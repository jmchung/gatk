package org.broadinstitute.hellbender.tools.spark.sv.discovery.prototype;

import htsjdk.samtools.SAMSequenceDictionary;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.broadcast.Broadcast;
import org.broadinstitute.hellbender.engine.datasources.ReferenceMultiSource;
import org.broadinstitute.hellbender.tools.spark.sv.discovery.AlignedContig;
import org.broadinstitute.hellbender.tools.spark.sv.discovery.AlignmentInterval;
import org.broadinstitute.hellbender.tools.spark.sv.utils.PairedStrandedIntervals;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVInterval;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SvCigarUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import scala.Tuple2;

import java.util.*;

/**
 * This deals with the special case where a contig's multiple (> 2) alignments has head and tail mapped to the same chr.
 * For the case where the head and tail mapped to different chromosome, we could decide to emit all BND records, but
 * that could be dealt with later.
 */
final class CpxVariantDetector implements VariantDetectorFromLocalAssemblyContigAlignments {

    @Override
    public void inferSvAndWriteVCF(final String vcfOutputFileName, final String sampleId, final JavaRDD<AlignedContig> localAssemblyContigs,
                                   final Broadcast<ReferenceMultiSource> broadcastReference,
                                   final Broadcast<SAMSequenceDictionary> broadcastSequenceDictionary,
                                   final Logger toolLogger) {
        
        // extract reference ordered jumping locations on reference

        // segment affected reference regions by jumping locations

        // make sense of event, i.e. provide interpretation, and extract corresponding alt haplotype

        // output VCF
    }

    // =================================================================================================================

    /**
     * Each pair of neighboring reference locations are meant to be used closed, i.e. [a, b].
     */
    private static List<SimpleInterval> extractReferenceOrdereredJumpLocations(final List<AlignmentInterval> alignmentConfiguration,
                                                                               final SAMSequenceDictionary refSequenceDictionary) {

        // A configuration has a series jumps on the reference as indicated by the chimeric alignments.

        return null;
    }

    /**
     * A jump has a starting and landing ref location.
     *
     * <p>
     * A jump can be:
     * <ul>
     *     <li>gapped--meaning a part of read is uncovered by neighboring AI's;</li>
     *     <li>connected--meaning neighboring--but not overlapping on the read--AI's leave no base on the read uncovered;</li>
     *     <li>retracting--meaning neighboring AI's overlap on the read, pointing to homology between their ref span</li>
     * </ul>
     * Among them, retracting jumps are the most difficult to deal with, mainly due to how to have a consistent
     * homology-yielding scheme.
     * </p>
     */
    private static final class Jump {
        enum JumpType {
            CONNECTED, GAPPED, RETRACTING
        }

        final JumpType type;
        PairedStrandedIntervals link;
        final SimpleInterval start;
        final SimpleInterval landing;

        Jump(final AlignmentInterval one, final AlignmentInterval two, final SAMSequenceDictionary refSequenceDictionary) {
            if (one.endInAssembledContig == two.startInAssembledContig - 1) {
                type = JumpType.CONNECTED;
            } else {
                type = one.endInAssembledContig > two.startInAssembledContig ? JumpType.RETRACTING : JumpType.GAPPED;

            }
            start = new SimpleInterval(one.referenceSpan.getContig(), one.referenceSpan.getEnd(), one.referenceSpan.getEnd());
            landing = new SimpleInterval(two.referenceSpan.getContig(), two.referenceSpan.getStart(), two.referenceSpan.getStart());
        }

        private static Tuple2<SimpleInterval, SimpleInterval> recomputeLocationsForRetractingJump(final AlignmentInterval one,
                                                                                                  final AlignmentInterval two,
                                                                                                  final SAMSequenceDictionary refSequenceDictionary) {
            return null;
        }
    }

    private static List<SVInterval> segmentReference(final List<SimpleInterval> jumpingLocations ) {
        return null;
    }

    /**
     * Splits input alignments into ones that are intended to be used for chimeric alignments and
     * ones that are not reliable for that purpose,
     * based on provided MQ and unique ref/read span length threshold.
     */
    private static Tuple2<List<AlignmentInterval>, List<AlignmentInterval>> classifyAlignments(final Iterator<AlignmentInterval> iterator,
                                                                                               final int mapQualThreshold,
                                                                                               final int uniqueRefSpanThreshold,
                                                                                               final int uniqueReadSpanThreshold) {

        final List<AlignmentInterval> good = new ArrayList<>(10); // 10 is a blunt guess
        final List<AlignmentInterval> bad  = new ArrayList<>(10);

        AlignmentInterval current = iterator.next();
        while ( iterator.hasNext() ) {
            final AlignmentInterval next = iterator.next();
            final AlnPairUniqueLength alnPairUniqueLength = new AlnPairUniqueLength(current, next);

            if (alignmentIsNonInformative(current.mapQual, mapQualThreshold,
                    alnPairUniqueLength.oneUniqRefLen, uniqueRefSpanThreshold, alnPairUniqueLength.oneUniqReadLen, uniqueReadSpanThreshold)) {
                bad.add(current);
                current = next;
            } else if (alignmentIsNonInformative(next.mapQual, mapQualThreshold,
                    alnPairUniqueLength.twoUniqRefLen, uniqueRefSpanThreshold, alnPairUniqueLength.twoUniqReadLen, uniqueReadSpanThreshold)) {
                bad.add(next);
            } else {
                good.add(current);
                current = next;
            }
        }

        return new Tuple2<>(good, bad);
    }

    private static boolean alignmentIsNonInformative(final int mapQ, final int mapqThresholdInclusive,
                                                     final int uniqRefSpanLen, final int uniqueRefSpanThreshold,
                                                     final int uniqReadSpanLen, final int uniqueReadSpanThreshold) {
        return mapQ < mapqThresholdInclusive
                || uniqRefSpanLen < uniqueRefSpanThreshold
                || uniqReadSpanLen < uniqueReadSpanThreshold;

    }

    /**
     * For representing unique reference span sizes and read consumption length values of two neighboring
     * alignment intervals of a particular contig.
     * Fields are mostly useful, for now, for filtering alignments.
     */
    private static final class AlnPairUniqueLength {
        final int oneUniqRefLen;
        final int oneUniqReadLen;
        final int twoUniqRefLen;
        final int twoUniqReadLen;

        AlnPairUniqueLength(final AlignmentInterval one, final AlignmentInterval two) {
            Utils.validateArg(one.startInAssembledContig <= two.startInAssembledContig,
                    "assumption that input alignments are order along read is violated");

            final int overlapOnRefSpan = AlignmentInterval.overlapOnRefSpan(one, two);
            final int overlapOnRead = AlignmentInterval.overlapOnContig(one, two);

            if (overlapOnRead == 0) {
                oneUniqRefLen = one.referenceSpan.size() - overlapOnRefSpan;
                twoUniqRefLen = two.referenceSpan.size() - overlapOnRefSpan;
                oneUniqReadLen = one.endInAssembledContig - one.startInAssembledContig + 1;
                twoUniqReadLen = two.endInAssembledContig - two.startInAssembledContig + 1;
            } else {
                // TODO: 10/16/17 hardclip offset
                final int i = SvCigarUtils.computeAssociatedDistOnRef(one.cigarAlong5to3DirectionOfContig, two.startInAssembledContig, overlapOnRead);
                final int j = SvCigarUtils.computeAssociatedDistOnRef(two.cigarAlong5to3DirectionOfContig, two.startInAssembledContig, overlapOnRead);
                oneUniqRefLen = one.referenceSpan.size() - Math.max(i, overlapOnRefSpan);
                twoUniqRefLen = two.referenceSpan.size() - Math.max(j, overlapOnRefSpan);
                oneUniqReadLen = one.endInAssembledContig - one.startInAssembledContig + 1 - overlapOnRead;
                twoUniqReadLen = two.endInAssembledContig - two.startInAssembledContig + 1 - overlapOnRead;
            }
        }
    }
}
