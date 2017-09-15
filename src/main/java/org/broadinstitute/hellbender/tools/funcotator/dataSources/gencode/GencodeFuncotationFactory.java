package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import com.google.common.annotations.VisibleForTesting;
import htsjdk.samtools.reference.ReferenceSequence;
import htsjdk.samtools.util.Locatable;
import htsjdk.tribble.Feature;
import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.funcotator.*;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * A factory to create {@link GencodeFuncotation}s.
 * Created by jonn on 8/30/17.
 */
public class GencodeFuncotationFactory extends DataSourceFuncotationFactory {

    //==================================================================================================================

    public GencodeFuncotationFactory() {}

    //==================================================================================================================

    /**
     * The window around splice sites to mark variants as {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification#SPLICE_SITE}.
     */
    final private int spliceSiteVariantWindowBases = 2;

    //==================================================================================================================

    @Override
    public List<String> getSupportedFuncotationFields() {
        return GencodeFuncotation.getSerializedFieldNames();
    }

    @Override
    public List<Funcotation> createFuncotations(final VariantContext variant, final ReferenceContext referenceContext, final List<Feature> featureList) {
        final List<Funcotation> funcotations = new ArrayList<>();

        // If we have features we need to annotate, go through them and create annotations:
        if ( featureList.size() > 0 ) {
            for ( final Allele altAllele : variant.getAlternateAlleles() ) {
                for ( final Feature feature : featureList ) {

                    // Get the kind of feature we want here:
                    if ( GencodeGtfGeneFeature.class.isAssignableFrom(feature.getClass()) ) {
                        funcotations.addAll(createFuncotations(variant, altAllele, (GencodeGtfGeneFeature) feature, referenceContext));
                    }

                    // NOTE: If we don't have any funcotations for this feature, it's OK.
                    //       However, this means that some other DataSourceFuncotationFactory must be producing a
                    //       funcotation for this variant.
                    //       For it is decreed that all variants must have a funcotation, even if that funcotation be
                    //       empty.
                    // TODO: Actually you may want to put another IGR creation here for now...  This may be a more difficult thing if we determine it in here.  There is no way to know if these are IGRs or simply not included in this particular data set.
                }
            }
        }
        else {
            // This is an IGR.
            funcotations.addAll( createIgrFuncotations(variant, referenceContext) );
        }

        return funcotations;
    }

    //==================================================================================================================

    /**
     * Creates a {@link List} of {@link GencodeFuncotation}s based on the given {@link VariantContext}, {@link Allele}, and {@link GencodeGtfGeneFeature}.
     * @param variant The variant to annotate.
     * @param altAllele The allele of the given variant to annotate.
     * @param gtfFeature The GTF feature on which to base annotations.
     * @return A {@link List} of {@link GencodeFuncotation}s for the given variant, allele and gtf feature.
     */
    private List<GencodeFuncotation> createFuncotations(final VariantContext variant, final Allele altAllele, final GencodeGtfGeneFeature gtfFeature, final ReferenceContext reference) {
        // For each applicable transcript, create an annotation.

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        // TODO: instead of getting the best one here, we do them all, then in the renderer we condense them into 1 annotation based on worst effect.
        // Get our "best" transcript:
        final int bestTranscriptIndex = getBestTranscriptIndex(gtfFeature, variant);
        if ( bestTranscriptIndex == -1 ) {
            throw new GATKException("Could not get a good transcript for the given feature: " + gtfFeature.toString());
        }
        final GencodeGtfTranscriptFeature transcript = gtfFeature.getTranscripts().remove(bestTranscriptIndex);

        final GencodeGtfFeature.GeneTranscriptType geneType = gtfFeature.getGeneType();

        // We only fully process protein-coding regions.
        // For other gene types, we do trivial processing and label them as either LINCRNA or RNA:
        // TODO: Add more types to Variant Classification to be more explicit and a converter function to go from gene type to variant classification.
        if ( geneType != GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING) {

            // Setup the "trivial" fields of the gencodeFuncotation:
            final GencodeFuncotation gencodeFuncotation = createGencodeFuncotationWithTrivialFieldsPopulated(variant, altAllele, gtfFeature, transcript);

            if ( geneType == GencodeGtfFeature.GeneTranscriptType.LINCRNA) {
                gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.LINCRNA);
            }
            else {
                gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.RNA);
            }
            gencodeFuncotations.add(gencodeFuncotation);
        }
        else {
            // Find the sub-feature of transcript that contains our variant:
            final GencodeGtfFeature containingSubfeature = getContainingGtfSubfeature(variant, transcript);

            // Determine what kind of region we're in and handle it in it's own way:
            if (containingSubfeature == null) {

                // We have an IGR variant
                gencodeFuncotations.add( createIgrFuncotation(altAllele) );

            } else if (GencodeGtfExonFeature.class.isAssignableFrom(containingSubfeature.getClass())) {

                // We have a coding region variant
                gencodeFuncotations.add( createCodingRegionFuncotation(variant, altAllele, gtfFeature, reference, transcript, (GencodeGtfExonFeature) containingSubfeature) );

            } else if (GencodeGtfUTRFeature.class.isAssignableFrom(containingSubfeature.getClass())) {

                // We have a UTR variant
                gencodeFuncotations.add( createUtrFuncotation(variant, altAllele, reference, gtfFeature, transcript, (GencodeGtfUTRFeature) containingSubfeature) );

            } else if (GencodeGtfTranscriptFeature.class.isAssignableFrom(containingSubfeature.getClass())) {

                // We have an intron variant
                gencodeFuncotations.add( createIntronFuncotation(variant, altAllele, gtfFeature, transcript) );

            } else {

                // Uh-oh!  Problemz.
                throw new GATKException("Unable to determine type of variant-containing subfeature: " + containingSubfeature.getClass().getName());
            }
        }

        return gencodeFuncotations;
    }

    /**
     * Create a {@link GencodeFuncotation} for a {@code variant} that occurs in a coding region in a {@code transcript}.
     * @param variant The {@link VariantContext} for which to create a {@link GencodeFuncotation}.
     * @param altAllele The {@link Allele} in the given {@code variant} for which to create a {@link GencodeFuncotation}.
     * @param gtfFeature The {@link GencodeGtfGeneFeature} in which the given {@code variant} occurs.
     * @param reference The {@link ReferenceContext} for the current data set.
     * @param transcript The {@link GencodeGtfTranscriptFeature} in which the given {@code variant} occurs.
     * @param exon The {@link GencodeGtfExonFeature} in which the given {@code variant} occurs.
     * @return A {@link GencodeFuncotation} containing information about the given {@code variant} given the corresponding {@code exon}.
     */
    private GencodeFuncotation createCodingRegionFuncotation(final VariantContext variant,
                                                             final Allele altAllele,
                                                             final GencodeGtfGeneFeature gtfFeature,
                                                             final ReferenceContext reference,
                                                             final GencodeGtfTranscriptFeature transcript,
                                                             final GencodeGtfExonFeature exon) {

        // Setup the "trivial" fields of the gencodeFuncotation:
        final GencodeFuncotation gencodeFuncotation = createGencodeFuncotationWithTrivialFieldsPopulated(variant, altAllele, gtfFeature, transcript);

        // Set the exon number:
        gencodeFuncotation.setTranscriptExon( exon.getExonNumber() );

        // Get the list of exons by their locations so we can use them to determine our location in the transcript and get
        // the transcript code itself:
        final List<? extends Locatable> exonPositionList =
                transcript.getExons().stream()
                        .filter(e -> (e.getCds() != null))
                        .map(GencodeGtfExonFeature::getCds)
                        .collect(Collectors.toList());

        // Set our transcript exon number:
        gencodeFuncotation.setTranscriptExon(exon.getExonNumber());

        // Set up our SequenceComparison object so we can calculate some useful fields more easily
        // These fields can all be set without knowing the alternate allele:
        final FuncotatorUtils.SequenceComparison sequenceComparison = createSequenceComparisonWithTrivialFieldsPopulated(variant, reference, transcript, exonPositionList);

        // Set our alternate allele:
        sequenceComparison.setAlternateAllele( altAllele.getBaseString() );

        // Set our stop position:
        sequenceComparison.setAlignedAlternateAlleleStop(
                FuncotatorUtils.getAlignedEndPosition(
                        sequenceComparison.getAlignedCodingSequenceAlleleStart(),
                        altAllele.length()
                )
        );

        // Get our alternate allele coding sequence:
        final String altCodingSequence = FuncotatorUtils.getAlternateCodingSequence(
                sequenceComparison.getWholeReferenceSequence().getBaseString(),
                sequenceComparison.getCodingSequenceAlleleStart(),
                variant.getReference(),
                altAllele);

        // Note we add 1 because substring ends are EXCLUSIVE:
        sequenceComparison.setAlignedAlternateAllele(
                altCodingSequence.substring(
                        sequenceComparison.getAlignedCodingSequenceAlleleStart() - 1, // We subtract 1 because we're 1-based.
                        sequenceComparison.getAlignedAlternateAlleleStop()
                )
        );

        // Set our alternate amino acid sequence:
        sequenceComparison.setAlternateAminoAcidSequence(
                FuncotatorUtils.createAminoAcidSequence(sequenceComparison.getAlignedAlternateAllele())
        );

        // Set our protein end position:
        sequenceComparison.setProteinChangeEndPosition(
                sequenceComparison.getProteinChangeStartPosition() + (sequenceComparison.getAlignedAlternateAllele().length() / 3) - 1 // We subtract 1 because we're 1-based.
        );

        // OK, now that we have our SequenceComparison object set up we can continue the annotation:

        // Set the DNA changes:
        gencodeFuncotation.setCodonChange(FuncotatorUtils.getCodonChangeString(sequenceComparison));
        gencodeFuncotation.setProteinChange(FuncotatorUtils.getProteinChangeString(sequenceComparison));
        gencodeFuncotation.setcDnaChange(FuncotatorUtils.getCodingSequenceChangeString(sequenceComparison));

        // Now all we have to do is decide what the variant classification type should be.

        gencodeFuncotation.setVariantClassification(
                getVariantClassification( variant, altAllele, gencodeFuncotation.getVariantType(), exon, sequenceComparison )
        );

        if ( gencodeFuncotation.getVariantClassification() == GencodeFuncotation.VariantClassification.SPLICE_SITE ) {
            gencodeFuncotation.setSecondaryVariantClassification(
                    getVariantClassificationForCodingRegion(variant, altAllele, gencodeFuncotation.getVariantType(), sequenceComparison )
            );
        }

        return gencodeFuncotation;
    }

    /**
     * Gets the {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification} of the given {@code altAllele} for the given {@code variant}.
     * @param variant The {@link VariantContext} to classify.
     * @param altAllele The {@link Allele} of the given {@code variant} to classify.
     * @param variantType The {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantType} of the given {@code variant}.
     * @param exon The {@link GencodeGtfExonFeature} in which the given {@code variant} occurs.
     * @param sequenceComparison The {@link org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils.SequenceComparison} for the given {@code variant}.
     * @return A {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification} based on the given {@code allele}, {@code variant}, {@code exon}, and {@code sequenceComparison}.
     */
    @VisibleForTesting
    GencodeFuncotation.VariantClassification getVariantClassification(final VariantContext variant,
                                                                      final Allele altAllele,
                                                                      final GencodeFuncotation.VariantType variantType,
                                                                      final GencodeGtfExonFeature exon,
                                                                      final FuncotatorUtils.SequenceComparison sequenceComparison ){

        final int startPos = sequenceComparison.getAlleleStart();
        final int endPos = sequenceComparison.getAlleleStart() + altAllele.length() - 1;

        GencodeFuncotation.VariantClassification varClass = null;

        boolean hasBeenClassified = false;

        // Check for non-stop first:
        if ( (exon.getStopCodon() != null) && (exon.getStopCodon().overlaps(variant)) ) {

            boolean foundStop = false;

            for (int i = 0 ; i < sequenceComparison.getAlignedAlternateAllele().length(); i+=3 ){
                final String codon = sequenceComparison.getAlignedAlternateAllele().substring(i, i+3);
                if (FuncotatorUtils.getEukaryoticAminoAcidByCodon(codon) == AminoAcid.STOP_CODON) {
                    foundStop = true;
                    break;
                }
            }

            if ( !foundStop ) {
                varClass = GencodeFuncotation.VariantClassification.NONSTOP;
                hasBeenClassified = true;
            }
        }

        // Now check all other cases:
        if ( !hasBeenClassified ) {

            if ((Math.abs(startPos - exon.getStart()) < spliceSiteVariantWindowBases) ||
                    (Math.abs(endPos - exon.getStart()) < spliceSiteVariantWindowBases) ||
                    (Math.abs(startPos - exon.getEnd()) < spliceSiteVariantWindowBases) ||
                    (Math.abs(endPos - exon.getEnd()) < spliceSiteVariantWindowBases)) {
                varClass = GencodeFuncotation.VariantClassification.SPLICE_SITE;
            }
            else if ((exon.getStartCodon() != null) && (exon.getStartCodon().overlaps(variant))) {
                switch (variantType) {
                    case INS:
                        varClass = GencodeFuncotation.VariantClassification.START_CODON_INS;
                        break;
                    case DEL:
                        varClass = GencodeFuncotation.VariantClassification.START_CODON_DEL;
                        break;
                    default:
                        varClass = GencodeFuncotation.VariantClassification.START_CODON_SNP;
                        break;
                }
            }
            else {
                varClass = getVariantClassificationForCodingRegion(variant, altAllele, variantType, sequenceComparison);
            }
        }

        return varClass;
    }

    /**
     * Get the {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification} for a given {@code variant}/{@code allele} in a coding region.
     * @param variant The {@link VariantContext} to classify.
     * @param altAllele The {@link Allele} of the given {@code variant} to classify.
     * @param variantType The {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantType} of the given {@code variant}.
     * @param sequenceComparison The {@link org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils.SequenceComparison} for the given {@code variant}.
     * @return A {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification} based on the given {@code allele}, {@code variant}, {@code exon}, and {@code sequenceComparison}.
     */
    private GencodeFuncotation.VariantClassification getVariantClassificationForCodingRegion(final VariantContext variant,
                                                                                             final Allele altAllele,
                                                                                             final GencodeFuncotation.VariantType variantType,
                                                                                             final FuncotatorUtils.SequenceComparison sequenceComparison) {
        final GencodeFuncotation.VariantClassification varClass;

        if (variantType == GencodeFuncotation.VariantType.INS) {
            if (FuncotatorUtils.isFrameshift(variant.getReference(), altAllele)) {
                varClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS;
            }
            else {
                varClass = GencodeFuncotation.VariantClassification.IN_FRAME_INS;
            }
        }
        else if (variantType == GencodeFuncotation.VariantType.DEL) {
            if (FuncotatorUtils.isFrameshift(variant.getReference(), altAllele)) {
                varClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL;
            }
            else {
                varClass = GencodeFuncotation.VariantClassification.IN_FRAME_DEL;
            }
        }
        else {
            // This is a SNP/MNP
            // We just check to see what the protein change is to check for MISSENSE/NONSENSE/SILENT:
            varClass = getVarClassFromEqualLengthCodingRegions( sequenceComparison );
        }

        return varClass;
    }

    /**
     * Gets the {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification} for a {@code variant} where the reference and alternate
     * alleles are the same length and the variant appears in a coding region.
     * This essentially compares the amino acid sequences of both alleles and returns a value based on the differences between them.
     * @return The {@link org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation.VariantClassification} corresponding to the given variant / reference allele / alternate allele.
     */
    private GencodeFuncotation.VariantClassification getVarClassFromEqualLengthCodingRegions(final FuncotatorUtils.SequenceComparison sequenceComparison) {

        GencodeFuncotation.VariantClassification varClass = GencodeFuncotation.VariantClassification.SILENT;

        boolean foundStop = false;

        for ( int i = 0; i < sequenceComparison.getAlternateAminoAcidSequence().length(); ++i ) {
            final char altAminoAcid = sequenceComparison.getAlternateAminoAcidSequence().charAt(i);

            if ( FuncotatorUtils.getAminoAcidByLetter(altAminoAcid) == AminoAcid.STOP_CODON ) {
                foundStop = true;
                break;
            }
            else if ( altAminoAcid != sequenceComparison.getReferenceAminoAcidSequence().charAt(i) ) {
                varClass = GencodeFuncotation.VariantClassification.MISSENSE;
            }
        }

        if ( foundStop ) {
            varClass = GencodeFuncotation.VariantClassification.NONSENSE;
        }

        return varClass;
    }

    /**
     * Create a {@link GencodeFuncotation} for a {@code variant} that occurs in an untranslated region in a given {@code transcript}.
     * @param variant The {@link VariantContext} for which to create a {@link GencodeFuncotation}.
     * @param altAllele The {@link Allele} in the given {@code variant} for which to create a {@link GencodeFuncotation}.
     * @param reference The {@link ReferenceContext} for the current data set.
     * @param gtfFeature The {@link GencodeGtfGeneFeature} in which the given {@code variant} occurs.
     * @param transcript The {@link GencodeGtfTranscriptFeature} in which the given {@code variant} occurs.
     * @param utr The {@link GencodeGtfUTRFeature} in which the given {@code variant} occurs.
     * @return A {@link GencodeFuncotation} containing information about the given {@code variant} given the corresponding {@code utr}.
     */
    private GencodeFuncotation createUtrFuncotation(final VariantContext variant,
                                                    final Allele altAllele,
                                                    final ReferenceContext reference,
                                                    final GencodeGtfGeneFeature gtfFeature,
                                                    final GencodeGtfTranscriptFeature transcript,
                                                    final GencodeGtfUTRFeature utr) {

        // Setup the "trivial" fields of the gencodeFuncotation:
        final GencodeFuncotation gencodeFuncotation = createGencodeFuncotationWithTrivialFieldsPopulated(variant, altAllele, gtfFeature, transcript);

        // Find which exon this UTR is in:
        for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
            if ( exon.contains( utr ) ) {
                gencodeFuncotation.setTranscriptExon( exon.getExonNumber() );
            }
        }

        // Set whether it's the 5' or 3' UTR:
        if ( is5PrimeUtr(utr, transcript) ) {

            // We're 5' to the coding region.
            // This means we need to check for de novo starts.

            // Get our coding sequence for this region:
            final List<Locatable> activeRegions = Collections.singletonList(utr);
            final String codingSequence = FuncotatorUtils.getCodingSequence(reference, activeRegions);
            final int codingStartPos = FuncotatorUtils.getStartPositionInTranscript(variant, activeRegions);

            //Check for de novo starts:
            if ( FuncotatorUtils.getEukaryoticAminoAcidByCodon(codingSequence.substring(codingStartPos, codingStartPos+3) )
                    == AminoAcid.METHIONINE ) {

                // We know we have a new start.
                // Is it in frame or out of frame?
                if ( FuncotatorUtils.isInFrameWithEndOfRegion(codingStartPos, codingSequence.length()) ) {
                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.DE_NOVO_START_IN_FRAME);
                }
                else {
                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.DE_NOVO_START_OUT_FRAME);
                }
            }
            else {
                gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.FIVE_PRIME_UTR);
            }
        }
        else {
            gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.THREE_PRIME_UTR);
        }

        return gencodeFuncotation;
    }

    /**
     * Create a {@link GencodeFuncotation} for a {@code variant} that occurs in an intron in the given {@code transcript}.
     * @param variant The {@link VariantContext} for which to create a {@link GencodeFuncotation}.
     * @param altAllele The {@link Allele} in the given {@code variant} for which to create a {@link GencodeFuncotation}.
     * @param gtfFeature The {@link GencodeGtfGeneFeature} in which the given {@code variant} occurs.
     * @param transcript The {@link GencodeGtfTranscriptFeature} in which the given {@code variant} occurs.
     * @return A {@link GencodeFuncotation} containing information about the given {@code variant} given the corresponding {@code transcript}.
     */
    private GencodeFuncotation createIntronFuncotation(final VariantContext variant,
                                                       final Allele altAllele,
                                                       final GencodeGtfGeneFeature gtfFeature,
                                                       final GencodeGtfTranscriptFeature transcript) {

        // Setup the "trivial" fields of the gencodeFuncotation:
        final GencodeFuncotation gencodeFuncotation = createGencodeFuncotationWithTrivialFieldsPopulated(variant, altAllele, gtfFeature, transcript);

        // Set as default INTRON variant classification:
        gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.INTRON);

        // Need to check if we're within the window for splice site variants:
        for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
            if (( Math.abs( exon.getStart() - variant.getStart() ) <= spliceSiteVariantWindowBases ) ||
                ( Math.abs( exon.getEnd() - variant.getStart() ) <= spliceSiteVariantWindowBases )) {
                gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.SPLICE_SITE);
                gencodeFuncotation.setSecondaryVariantClassification(GencodeFuncotation.VariantClassification.INTRON);
            }
        }

        return gencodeFuncotation;
    }

    /**
     * Get the subfeature contained in {@code transcript} that contains the given {@code variant}.
     * The returned subfeature will be of type {@link GencodeGtfFeature} with concrete type based on the type of region
     * in which the variant is found:
     *      Found in coding region -> {@link GencodeGtfExonFeature}
     *      Found in UTR ->{@link GencodeGtfUTRFeature}
     *      Found in intron ->{@link GencodeGtfTranscriptFeature}
     *      Not Found in transcript ->{@code null}
     * @param variant A {@link VariantContext} of which to determine the containing subfeature.
     * @param transcript A {@link GencodeGtfTranscriptFeature} in which to find the subfeature containing the given {@code variant}.
     * @return The {@link GencodeGtfFeature} corresponding to the subfeature of {@code transcript} in which the given {@code variant} was found.
     */
    private GencodeGtfFeature getContainingGtfSubfeature(final VariantContext variant, final GencodeGtfTranscriptFeature transcript) {

        boolean determinedRegionAlready = false;
        GencodeGtfFeature subFeature = null;

        if ( transcript.contains(variant) ) {
            if ( transcript.getUtrs().size() > 0 ) {
                for ( final GencodeGtfUTRFeature utr : transcript.getUtrs() ) {
                    if ( utr.contains(variant) ) {
                        subFeature = utr;
                        determinedRegionAlready = true;
                    }
                }
            }

            if ( !determinedRegionAlready ) {
                for (final GencodeGtfExonFeature exon : transcript.getExons()) {
                    final GencodeGtfCDSFeature cds = exon.getCds();
                    if ((cds != null) && (cds.contains(variant))) {
                        subFeature = exon;
                        determinedRegionAlready = true;
                    }
                }
            }

            if ( !determinedRegionAlready ) {
                subFeature = transcript;
            }
        }

        return subFeature;
    }

    /**
     * Creates a {@link org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils.SequenceComparison} object with the trivial fields populated.
     * @param variant The {@link VariantContext} for the current variant.
     * @param reference The {@link ReferenceContext} for the current sample set.
     * @param transcript The {@link GencodeGtfTranscriptFeature} for the current gene feature / alt allele.
     * @param exonPositionList A {@link List} of {@link htsjdk.samtools.util.Locatable} objects representing exon positions in the transcript.
     * @return A trivially populated {@link org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils.SequenceComparison} object.
     */
    private FuncotatorUtils.SequenceComparison createSequenceComparisonWithTrivialFieldsPopulated(final VariantContext variant,
                                                                                                  final ReferenceContext reference,
                                                                                                  final GencodeGtfTranscriptFeature transcript,
                                                                                                  final List<? extends htsjdk.samtools.util.Locatable> exonPositionList) {

        final FuncotatorUtils.SequenceComparison sequenceComparison = new FuncotatorUtils.SequenceComparison();

        // Get the contig:
        sequenceComparison.setContig(variant.getContig());

        // Get the reference sequence in the coding region as described by the given exonPositionList:
        sequenceComparison.setWholeReferenceSequence(new ReferenceSequence(transcript.getTranscriptId(),transcript.getStart(),FuncotatorUtils.getCodingSequence(reference, exonPositionList).getBytes()));

        // Get the ref allele:
        sequenceComparison.setReferenceAllele(variant.getReference().getBaseString());

        // Get the allele genomic start position:
        sequenceComparison.setAlleleStart(variant.getStart());

        // Get the allele transcript start position:
        sequenceComparison.setTranscriptAlleleStart(variant.getStart() - transcript.getStart());

        // Get the coding region start position (in the above computed reference coding region):
        sequenceComparison.setCodingSequenceAlleleStart(FuncotatorUtils.getStartPositionInTranscript(variant, exonPositionList));

        // Get the in-frame start position of the codon containing the given variant:
        sequenceComparison.setAlignedCodingSequenceAlleleStart(FuncotatorUtils.getAlignedPosition(sequenceComparison.getCodingSequenceAlleleStart()));

        // Get the in-frame stop position of the codon containing the given variant:
        sequenceComparison.setAlignedReferenceAlleleStop(FuncotatorUtils.getAlignedEndPosition(sequenceComparison.getAlignedCodingSequenceAlleleStart(),
                variant.getReference().length() ));

        // Get the in-frame/codon-aligned region containing the reference allele:
        sequenceComparison.setAlignedReferenceAllele(
                sequenceComparison.getWholeReferenceSequence().getBaseString().substring(
                        sequenceComparison.getAlignedCodingSequenceAlleleStart() - 1, // Subtract 1 because we're 1-based.
                        sequenceComparison.getAlignedReferenceAlleleStop()
                )
        );

        // Get the amino acid sequence of the reference allele:
        sequenceComparison.setReferenceAminoAcidSequence(
                FuncotatorUtils.createAminoAcidSequence( sequenceComparison.getAlignedReferenceAllele() )
        );

        // Get the starting protein position of this variant.
        sequenceComparison.setProteinChangeStartPosition(
                (sequenceComparison.getAlignedCodingSequenceAlleleStart() / 3) + 1 // Add 1 because we're 1-based.
        );

        return sequenceComparison;
    }

    /**
     * Creates a Gencode Funcotation with all trivial fields populated.
     * @param variant The {@link VariantContext} for the current variant.
     * @param altAllele The alternate {@link Allele} we are currently annotating.
     * @param gtfFeature The current {@link GencodeGtfGeneFeature} read from the input feature file.
     * @param transcript The current {@link GencodeGtfTranscriptFeature} containing our {@code altAllele}.
     * @return A trivially populated {@link GencodeFuncotation} object.
     */
    private GencodeFuncotation createGencodeFuncotationWithTrivialFieldsPopulated(final VariantContext variant,
                                                                                  final Allele altAllele,
                                                                                  final GencodeGtfGeneFeature gtfFeature,
                                                                                  final GencodeGtfTranscriptFeature transcript) {
        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        gencodeFuncotation.setHugoSymbol( gtfFeature.getGeneName() );
        gencodeFuncotation.setNcbiBuild( gtfFeature.getUcscGenomeVersion() );
        gencodeFuncotation.setChromosome( gtfFeature.getChromosomeName() );

        gencodeFuncotation.setStart(variant.getStart());

        // The end position is inclusive, so we need to make sure we don't double-count the start position (so we subtract 1):
        gencodeFuncotation.setEnd(variant.getStart() + altAllele.length() - 1);

        gencodeFuncotation.setVariantType( getVariantType(variant.getReference(), altAllele) );
        gencodeFuncotation.setRefAllele( variant.getReference().getBaseString() );
        gencodeFuncotation.setTumorSeqAllele1( altAllele.getBaseString() );
        gencodeFuncotation.setTumorSeqAllele2( altAllele.getBaseString() );

        gencodeFuncotation.setGenomeChange(getGenomeChangeString(variant, altAllele, gtfFeature));
        gencodeFuncotation.setAnnotationTranscript( transcript.getTranscriptId() );
        gencodeFuncotation.setTranscriptStrand( transcript.getGenomicStrand().toString() );
        gencodeFuncotation.setTranscriptPos( variant.getStart() - transcript.getStart() );

        gencodeFuncotation.setOtherTranscripts(
                gtfFeature.getTranscripts().stream().map(GencodeGtfTranscriptFeature::getTranscriptId).collect(Collectors.toList())
        );

        return gencodeFuncotation;
    }

    /**
     * Determines if the given UTR is 3' or 5' of the given transcript.
     * Assumes the UTR is part of the given transcript.
     * @param utr The {@link GencodeGtfUTRFeature} to check for relative location in the given {@link GencodeGtfTranscriptFeature}.
     * @param transcript The {@link GencodeGtfTranscriptFeature} in which to check for the given {@code utr}.
     * @return {@code true} if the given {@code utr} is 5' for the given {@code transcript}; {@code false} otherwise.
     */
    private boolean is5PrimeUtr(final GencodeGtfUTRFeature utr, final GencodeGtfTranscriptFeature transcript) {
        boolean isBefore = true;
        if ( transcript.getGenomicStrand() == GencodeGtfFeature.GenomicStrand.FORWARD ) {
            for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
                if ( exon.getStart() < utr.getStart()) {
                    isBefore = false;
                    break;
                }
            }
        }
        else {
            for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
                if ( exon.getStart() > utr.getStart()) {
                    isBefore = false;
                    break;
                }
            }
        }

        return isBefore;
    }

    /**
     * Creates a string representing the genome change given the variant, allele, and gene feature for this variant.
     * @param variant {@link VariantContext} of which to create the change.
     * @param altAllele {@link Allele} representing the alternate allele for this variant.
     * @param gtfFeature {@link GencodeGtfGeneFeature} corresponding to this variant.
     * @return A short {@link String} representation of the genomic change for the given variant, allele, and feature.
     */
    private String getGenomeChangeString(final VariantContext variant, final Allele altAllele, final GencodeGtfGeneFeature gtfFeature) {
        return "g." + gtfFeature.getChromosomeName() +
                ":" + variant.getStart() +
                variant.getReference().getBaseString() + ">" + altAllele.getBaseString();
    }

    /**
     * Return the index of the "best" transcript in this gene.
     * @param geneFeature A {@link GencodeGtfGeneFeature} from which to get the index of the "best" transcript.
     * @param variant The {@link VariantContext} for which we want to get the best index.
     * @return The index of the "best" {@link GencodeGtfTranscriptFeature} in the given {@link GencodeGtfGeneFeature}.  Returns -1 if no transcript is present.
     */
    private int getBestTranscriptIndex(final GencodeGtfGeneFeature geneFeature, final VariantContext variant) {
        if ( geneFeature.getTranscripts().size() == 0 ) {
            return -1;
        }

        for ( int i = 0 ; i < geneFeature.getTranscripts().size() ; ++i ) {
            if ( geneFeature.getTranscripts().get(i).getGenomicPosition().overlaps(variant) ) {
                return i;
            }
        }

        // Oops... we didn't find anything.
        return -1;
    }

    /**
     * Creates a {@link List} of {@link GencodeFuncotation}s based on the given {@link VariantContext} with type
     * {@link GencodeFuncotation.VariantClassification#IGR}.
     * @param variant The variant to annotate.
     * @param reference The reference against which to compare the given variant.
     * @return A list of IGR annotations for the given variant.
     */
    private List<GencodeFuncotation> createIgrFuncotations(final VariantContext variant, final ReferenceContext reference) {
        // for each allele, create an annotation.

        // TODO: NEED TO FIX THIS LOGIC TO INCLUDE MORE INFO!

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        for ( final Allele allele : variant.getAlternateAlleles() ) {
            gencodeFuncotations.add( createIgrFuncotation(allele) );
        }

        return gencodeFuncotations;
    }

    /**
     * Creates a {@link GencodeFuncotation}s based on the given {@link Allele} with type
     * {@link GencodeFuncotation.VariantClassification#IGR}.
     * @param altAllele The alternate allele to use for this funcotation.
     * @return An IGR funcotation for the given allele.
     */
    private GencodeFuncotation createIgrFuncotation(final Allele altAllele){
        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        gencodeFuncotation.setVariantClassification( GencodeFuncotation.VariantClassification.IGR );
        gencodeFuncotation.setTumorSeqAllele1( altAllele.getBaseString() );
        gencodeFuncotation.setTumorSeqAllele2( altAllele.getBaseString() );

        return gencodeFuncotation;
    }

    /**
     * Determines the variant type based on the given reference allele and alternate allele.
     * @param refAllele The reference {@link Allele} for this variant.
     * @param altAllele The alternate {@link Allele} for this variant.
     * @return A {@link GencodeFuncotation.VariantType} representing the variation type between the given reference and alternate {@link Allele}.
     */
    private GencodeFuncotation.VariantType getVariantType( final Allele refAllele, final Allele altAllele ) {

        if ( altAllele.length() > refAllele.length() ) {
            return GencodeFuncotation.VariantType.INS;
        }
        else if (altAllele.length() < refAllele.length()) {
            return GencodeFuncotation.VariantType.DEL;
        }
        else {
            // We know they are the same length, now we just need to check one of them:
            switch (refAllele.length()) {
                case 1:  return GencodeFuncotation.VariantType.SNP;
                case 2:  return GencodeFuncotation.VariantType.DNP;
                case 3:  return GencodeFuncotation.VariantType.TNP;
                default: return GencodeFuncotation.VariantType.ONP;
            }
        }
    }
}
