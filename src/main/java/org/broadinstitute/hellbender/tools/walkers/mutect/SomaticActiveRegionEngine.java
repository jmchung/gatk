package org.broadinstitute.hellbender.tools.walkers.mutect;

import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.util.Interval;
import htsjdk.samtools.util.Locatable;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.vcf.VCFConstants;
import org.apache.commons.math3.util.Pair;
import org.broadinstitute.hellbender.engine.AlignmentContext;
import org.broadinstitute.hellbender.engine.AssemblyRegionEvaluator;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.utils.MathUtils;
import org.broadinstitute.hellbender.utils.QualityUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.activityprofile.ActivityProfileState;
import org.broadinstitute.hellbender.utils.pileup.PileupElement;
import org.broadinstitute.hellbender.utils.pileup.ReadPileup;
import org.broadinstitute.hellbender.utils.realignmentfilter.Realigner;

import java.util.List;
import java.util.Optional;

public class SomaticActiveRegionEngine implements AssemblyRegionEvaluator {
    public static final int INDEL_START_QUAL = 30;
    public static final int INDEL_CONTINUATION_QUAL = 10;
    public static final double MAX_ALT_FRACTION_IN_NORMAL = 0.3;
    public static final int MAX_NORMAL_QUAL_SUM = 100;

    // if qual sum exceeds this amount, no need to continue realigning alt reads
    public static final int REALIGNMENT_QUAL_SUM_THRESHOLD = 80;

    public static final int MAX_REALIGNMENT_FAILS = 2;
    public static final int MINIMUM_BASE_QUALITY = 6;   // for active region determination

    final M2ArgumentCollection MTAC;
    final SAMFileHeader header;
    private final Optional<Realigner> realigner;
    private final boolean hasNormal;

    public SomaticActiveRegionEngine(final M2ArgumentCollection MTAC, final SAMFileHeader header) {
        this.MTAC = MTAC;
        hasNormal = MTAC.normalSampleName != null;
        this.header = header;
        realigner = MTAC.realignmentFilterArgumentCollection.bwaMemIndexImage == null ? Optional.empty() :
                Optional.of(new Realigner(MTAC.realignmentFilterArgumentCollection, header));
    }

    @Override
    public ActivityProfileState isActive(final AlignmentContext context, final ReferenceContext ref, final FeatureContext featureContext) {
        final byte refBase = ref.getBase();
        final SimpleInterval refInterval = ref.getInterval();

        if( context == null || context.getBasePileup().isEmpty() ) {
            return new ActivityProfileState(refInterval, 0.0);
        }

        final ReadPileup pileup = context.getBasePileup();
        final ReadPileup tumorPileup = pileup.getPileupForSample(MTAC.tumorSampleName, header);
        final Pair<Integer, Double> tumorAltCountAndQualSum = altCountAndQualSum(tumorPileup, refBase);
        final int tumorAltCount = tumorAltCountAndQualSum.getFirst();
        final int tumorRefCount = tumorPileup.size() - tumorAltCount;

        final double tumorLog10Odds = -QualityUtils.qualToErrorProbLog10(tumorAltCountAndQualSum.getSecond()) +
                MathUtils.log10Factorial(tumorAltCount) + MathUtils.log10Factorial(tumorRefCount) - MathUtils.log10Factorial(tumorPileup.size() + 1);

        if (tumorLog10Odds < MTAC.initialTumorLodThreshold) {
            return new ActivityProfileState(refInterval, 0.0);
        } else if (hasNormal) {
            final ReadPileup normalPileup = pileup.getPileupForSample(MTAC.normalSampleName, header);
            final Pair<Integer, Double> normalAltCountAndQualSum = altCountAndQualSum(normalPileup, refBase);
            final int normalAltCount = normalAltCountAndQualSum.getFirst();
            final double normalQualSum = normalAltCountAndQualSum.getSecond();
            if (normalAltCount > normalPileup.size() * MAX_ALT_FRACTION_IN_NORMAL && normalQualSum > MAX_NORMAL_QUAL_SUM) {
                return new ActivityProfileState(refInterval, 0.0);
            }
        } else {
            final List<VariantContext> germline = featureContext.getValues(MTAC.germlineResource, refInterval);
            if (!germline.isEmpty() && germline.get(0).getAttributeAsDoubleList(VCFConstants.ALLELE_FREQUENCY_KEY, 0.0).get(0) > MTAC.maxPopulationAlleleFrequency) {
                return new ActivityProfileState(refInterval, 0.0);
            }
        }

        if (!MTAC.genotypePonSites && !featureContext.getValues(MTAC.pon, new SimpleInterval(context.getContig(), (int) context.getPosition(), (int) context.getPosition())).isEmpty()) {
            return new ActivityProfileState(refInterval, 0.0);
        }

        return new ActivityProfileState( refInterval, 1.0, ActivityProfileState.Type.NONE, null);
    }

    private static int getCurrentOrFollowingIndelLength(final PileupElement pe) {
        return pe.isDeletion() ? pe.getCurrentCigarElement().getLength() : pe.getLengthOfImmediatelyFollowingIndel();
    }

    private static double indelQual(final int indelLength) {
        return INDEL_START_QUAL + (indelLength - 1) * INDEL_CONTINUATION_QUAL;
    }

    private Pair<Integer, Double> altCountAndQualSum(final ReadPileup pileup, final byte refBase) {

        int altCount = 0;
        double qualSum = 0;
        int realignmentFailCount = 0;

        Interval supposedRealignmentLocation = null;
        for (final PileupElement pe : pileup) {
            if (pe.getRead().getMappingQuality() == 0) {
                realignmentFailCount++;
                continue;
            }
            final double altQual = altQuality(pe, refBase);
            if (altQual > 0) {
                if (supposedRealignmentLocation == null) {
                    final Locatable loc = pileup.getLocation();
                    final Interval interval = new Interval(loc.getContig(), loc.getStart(), loc.getEnd());
                    supposedRealignmentLocation = realigner.isPresent() ? realigner.get().getRealignemntCoordinates(interval) : interval;
                }

                if (!realigner.isPresent() || qualSum > REALIGNMENT_QUAL_SUM_THRESHOLD || realigner.get().mapsToSupposedLocation(pe.getRead(), supposedRealignmentLocation)) {
                    qualSum += altQual;
                    altCount++;
                } else {
                    pe.getRead().setMappingQuality(0);
                    realignmentFailCount++;
                    if (realignmentFailCount > MAX_REALIGNMENT_FAILS) {
                        return new Pair<>(0, 0.0);
                    }
                }
            }
        }

        return new Pair<>(altCount, qualSum);
    }

    private static double altQuality(final PileupElement pe, final byte refBase) {
        final int indelLength = getCurrentOrFollowingIndelLength(pe);
        if (indelLength > 0) {
            return indelQual(indelLength);
        } else if (isNextToUsefulSoftClip(pe)) {
            return indelQual(1);
        } else if (pe.getBase() != refBase && pe.getQual() > MINIMUM_BASE_QUALITY) {
            return pe.getQual();
        } else {
            return 0;
        }
    }

    // check that we're next to a soft clip that is not due to a read that got out of sync and ended in a bunch of BQ2's
    // we only need to check the next base's quality
    private static boolean isNextToUsefulSoftClip(final PileupElement pe) {
        final int offset = pe.getOffset();
        return pe.getQual() > MINIMUM_BASE_QUALITY &&
                ((pe.isBeforeSoftClip() && pe.getRead().getBaseQuality(offset + 1) > MINIMUM_BASE_QUALITY)
                        || (pe.isAfterSoftClip() && pe.getRead().getBaseQuality(offset - 1) > MINIMUM_BASE_QUALITY));
    }

}
