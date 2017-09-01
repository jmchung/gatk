package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import htsjdk.tribble.Feature;
import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.funcotator.AminoAcid;
import org.broadinstitute.hellbender.tools.funcotator.DataSourceFuncotationFactory;
import org.broadinstitute.hellbender.tools.funcotator.Funcotation;
import org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A factory to create {@link GencodeFuncotation}s.
 * Created by jonn on 8/30/17.
 */
public class GencodeFuncotationFactory extends DataSourceFuncotationFactory {

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

        final SimpleInterval variantPosition = new SimpleInterval(variant.getContig(), variant.getStart(), variant.getEnd());

        // Get our transcript:
        final int bestTranscriptIndex = getBestTranscriptIndex(gtfFeature, variant);
        if ( bestTranscriptIndex == -1 ) {
            throw new GATKException("Could not get a good transcript for the given feature: " + gtfFeature.toString());
        }
        final GencodeGtfTranscriptFeature transcript = gtfFeature.getTranscripts().remove(bestTranscriptIndex);


//        private GencodeFuncotation.VariantClassification variantClassification;
//
//        private String                  genomeChange;
//        private int                     transcriptExon;
//        private int                     transcriptPos;
//        private String                  cDnaChange;
//        private String                  codonChange;
//        private String                  proteinChange;

        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        // Set the "trivial" fields of the funcotation up top:
        gencodeFuncotation.setHugoSymbol( gtfFeature.getGeneName() );
        gencodeFuncotation.setChromosome( gtfFeature.getChromosomeName() );
        gencodeFuncotation.setOtherTranscripts(
                gtfFeature.getTranscripts().stream().map(GencodeGtfTranscriptFeature::getTranscriptId).collect(Collectors.toList())
        );
        gencodeFuncotation.setStart(variant.getStart());
        gencodeFuncotation.setEnd(variant.getStart() + altAllele.length());
        gencodeFuncotation.setVariantType( getVariantType(variant.getReference(), altAllele) );
        gencodeFuncotation.setRefAllele( variant.getReference().getBaseString() );
        gencodeFuncotation.setTumorSeqAllele1( altAllele.getBaseString() );
        gencodeFuncotation.setTumorSeqAllele2( altAllele.getBaseString() );
        gencodeFuncotation.setAnnotationTranscript( transcript.getTranscriptId() );
        if ( transcript.getGenomicStrand() == GencodeGtfFeature.GenomicStrand.FORWARD ) {
            gencodeFuncotation.setTranscriptStrand("3'");
        }
        else {
            gencodeFuncotation.setTranscriptStrand("5'");
        }


//        // Go through our sub-features and determine what the VariantClassification is:
//
//        // Find our Exon:
//        for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
//
//            // Set our Exon fields:
//            gencodeFuncotation.setFeatureType( "Exon" );
//            gencodeFuncotation.setFeature( exon.getExonId() );
//            gencodeFuncotation.setExon( exon.getExonNumber() );
//
//            if ( exon.getGenomicPosition().overlaps(variantPosition) ) {
//                // Find the variant in this exon:
//                if ((exon.getStartCodon() != null) && exon.getStartCodon().getGenomicPosition().overlaps(variantPosition)) {
//                    // It's a start codon variant.
//
//                    if ( variant.getType() == VariantContext.Type.SNP ) {
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_SNP );
//                    }
//                    else if ( variant.getType() == VariantContext.Type.MNP ) {
//                        // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_INS );
//                    }
//                    else if ( variant.getType() == VariantContext.Type.INDEL ) {
//                        // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_INS );
//                    }
//                    else {
//                        // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_DEL );
//                    }
//
//                } else if ((exon.getCds() != null) && exon.getCds().getGenomicPosition().overlaps(variantPosition)) {
//                    // It's a 'normal' variant in the coding region.
//
//                    final GencodeFuncotation.VariantClassification variantClass;
//
//                    switch (variant.getType()) {
//                        case SNP:
//                            variantClass = getVariantClassificationSnp(variant, altAllele, reference, transcript);
//                            break;
//                        case MIXED:
//                        case MNP:
//                        case SYMBOLIC:
//                        case INDEL:
//                            if ( isFrameshift(variant.getReference(), altAllele) ) {
//                                // TODO: THIS IS ALMOST CERTAINLY WRONG:
//                                if ( altAllele.length() < variant.getReference().length()) {
//                                    variantClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL;
//                                }
//                                else if ( altAllele.length() > variant.getReference().length()) {
//                                    variantClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS;
//                                }
//                                else {
//                                    variantClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_SUB;
//                                }
//                            }
//                            else {
//                                if ( altAllele.length() < variant.getReference().length()) {
//                                    variantClass = GencodeFuncotation.VariantClassification.IN_FRAME_DEL;
//                                }
//                                else if ( altAllele.length() > variant.getReference().length()) {
//                                    variantClass = GencodeFuncotation.VariantClassification.IN_FRAME_INS;
//                                }
//                                else {
//                                    variantClass = getVariantClassificationSnp(variant, altAllele, reference, transcript);
//                                }
//                            }
//                            break;
//                        case NO_VARIATION:
//                        default:
//                            variantClass = GencodeFuncotation.VariantClassification.SILENT;
//                            break;
//                    }
//
//                    gencodeFuncotation.setClassification( variantClass );
//
//                    // Get the CDS position:
//                    final int cdsPosition = variantPosition.getStart() - exon.getCds().getGenomicPosition().getStart();
//                    gencodeFuncotation.setCdsPosition( cdsPosition );
//                    gencodeFuncotation.setProteinPosition( (int)Math.ceil(cdsPosition/3.0) );
//
//                } else {
//                    // Must be a stop_codon variant.
//
//                    if ( variant.getType() == VariantContext.Type.SNP ) {
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_SNP );
//                    }
//                    else if ( variant.getType() == VariantContext.Type.MNP ) {
//                        // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_INS );
//                    }
//                    else if ( variant.getType() == VariantContext.Type.INDEL ) {
//                        // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_INS );
//                    }
//                    else {
//                        // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
//                        gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_DEL );
//                    }
//
//                }
//
//                gencodeFuncotations.add(gencodeFuncotation);
//                hasHandledVariant = true;
//            }
//        }
//
//        // Find our UTR:
//        for ( final GencodeGtfUTRFeature utr : transcript.getUtrs() ) {
//            if ( utr.getGenomicPosition().overlaps(variantPosition) ) {
//                // We need to determine if this is a 3' or 5' UTR variation
//
//                // There are 2 UTRs.  One is at the front (3'), the other at the back (5'):
//                if ( utr.getGenomicStartLocation() == transcript.getStart() ) {
//                    gencodeFuncotation.setClassification(GencodeFuncotation.VariantClassification.THREE_PRIME_UTR);
//                }
//                else {
//                    gencodeFuncotation.setClassification(GencodeFuncotation.VariantClassification.FIVE_PRIME_UTR);
//                }
//
//                gencodeFuncotations.add(gencodeFuncotation);
//                hasHandledVariant = true;
//            }
//        }
//
//        // NOTE: We do not have to handle the Selenocysteines differently.
//        // They will be contained in CDS regions and will be handled there.
//
//        // Must be an intron:
//        gencodeFuncotation.setClassification(GencodeFuncotation.VariantClassification.INTRON);



        gencodeFuncotations.add(gencodeFuncotation);
        return gencodeFuncotations;
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
     * Creates a {@link List} of {@link GencodeFuncotation}s based on the given {@link VariantContext}.
     * These are most likely to be {@link GencodeFuncotation.VariantClassification#IGR}, but could also be
     * {@link GencodeFuncotation.VariantClassification#DE_NOVO_START_OUT_FRAME}
     * @param variant The variant to annotate.
     * @param reference The reference against which to compare the given variant.
     * @return A list of IGR annotations for the given variant.
     */
    private List<GencodeFuncotation> createIgrFuncotations(final VariantContext variant, final ReferenceContext reference) {
        // for each allele, create an annotation.

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        final int windowSizeInBases = 3;

        reference.setWindow(windowSizeInBases,windowSizeInBases);
        final String referenceSequence = new String( reference.getBases() );

        for ( final Allele allele : variant.getAlternateAlleles() ) {
            gencodeFuncotations.add(
                    createIgrAnnotationForAllele(allele,
                            referenceSequence.substring(0,windowSizeInBases) +
                            allele.getBaseString() +
                            referenceSequence.substring(windowSizeInBases + variant.getReference().length()))
            );
        }

        return gencodeFuncotations;
    }

    /**
     * Create a {@link GencodeFuncotation} representing the intergenic region variant given by {@code allele} and the {@code reference}.
     * @param altAllele An alternate {@link Allele} from which to create a {@link GencodeFuncotation}.
     * @param newSequence A {@link String} representing the new genetic sequence with the given alternate allele.
     * @return A {@link GencodeFuncotation} corresponding to an IGR variant given an allele and reference.
     */
    private GencodeFuncotation createIgrAnnotationForAllele(final Allele altAllele, final String newSequence) {

        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        // Determine if we're an IGR or a DE_NOVO_START:
        GencodeFuncotation.VariantClassification classification = GencodeFuncotation.VariantClassification.IGR;

        if ( newSequence.contains(AminoAcid.START_CODON.getCodons()[0])) {
            classification = GencodeFuncotation.VariantClassification.DE_NOVO_START_OUT_FRAME;
        }

        gencodeFuncotation.setVariantClassification( classification );
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
    private GencodeFuncotation.VariantType getVariantType(final Allele refAllele, final Allele altAllele ) {

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

     /**
     * Calculates the offset required to get the aligned codon start for the given variant.
     * @param variant {@link VariantContext} to align.
     * @param transcript {@link GencodeGtfTranscriptFeature} against which to align the given {@link VariantContext}
     * @return An offset which when subtracted from the variant start position will give the start of the codon containing the first base of the variant.
     */
    private int calculateOffsetToTranscriptAlignment(final VariantContext variant, final GencodeGtfTranscriptFeature transcript) {
        if ( !transcript.getGenomicPosition().overlaps(variant) ) {
            return 0;
        }

        return ((variant.getStart() - transcript.getStart()) % 3);
    }

    /**
     * Gets a {@link GencodeFuncotation.VariantClassification} based on a given {@code variant} and contextual information.
     * @param variant The {@link VariantContext} to classify.
     * @param allele The {@link Allele} of the {@link VariantContext} to classify.
     * @param reference The {@link ReferenceContext} against which to classify the {@link VariantContext}.
     * @param transcript The transcript against which to classify the {@link VariantContext}.
     * @return A {@link GencodeFuncotation.VariantClassification} corresponding to the given {@code variant} and contextual information.
     */
    private GencodeFuncotation.VariantClassification getVariantClassificationSnp(final VariantContext variant,
                                                                                 final Allele allele,
                                                                                 final ReferenceContext reference,
                                                                                 final GencodeGtfTranscriptFeature transcript) {
        final GencodeFuncotation.VariantClassification variantClass;

        // the reference is aligned exactly to the variant.
        // We need to line up the reference and the variant to the transcript.
        final int variantStartOffset = calculateOffsetToTranscriptAlignment(variant, transcript);

        // TODO: This is insufficient because of splice sites:
        reference.setWindow(variant.getStart() - variantStartOffset, 3);
        final byte[] rawBases = reference.getBases();
        final byte[] bases = new byte[] { rawBases[0], rawBases[1], rawBases[2] };

        final byte[] variantBases = bases.clone();
        variantBases[-variantStartOffset] = allele.getBases()[0];

        final AminoAcid referenceAminoAcid = FuncotatorUtils.getEukaryoticAminoAcidByCodon( new String(bases) );
        final AminoAcid variantAminoAcid = FuncotatorUtils.getEukaryoticAminoAcidByCodon( new String(variantBases) );

        if ( variantAminoAcid == null ) {
            variantClass = GencodeFuncotation.VariantClassification.NONSENSE;
        }
        else {
            if ( referenceAminoAcid == variantAminoAcid ) {
                variantClass = GencodeFuncotation.VariantClassification.SILENT;
            }
            else {
                variantClass = GencodeFuncotation.VariantClassification.MISSENSE;
            }
        }

        return variantClass;
    }
}
