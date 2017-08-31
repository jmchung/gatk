package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import htsjdk.tribble.Feature;
import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.tools.funcotator.AminoAcid;
import org.broadinstitute.hellbender.tools.funcotator.DataSourceFuncotationFactory;
import org.broadinstitute.hellbender.tools.funcotator.Funcotation;
import org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;

import java.util.ArrayList;
import java.util.List;

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
            for ( final Allele allele : variant.getAlternateAlleles() ) {
                for ( final Feature feature : featureList ) {

                    // Get the kind of feature we want here:
                    if ( GencodeGtfGeneFeature.class.isAssignableFrom(feature.getClass()) ) {
                        funcotations.addAll(createFuncotations(variant, allele, (GencodeGtfGeneFeature) feature, referenceContext));
                    }

                    // NOTE: If we don't have any funcotations for this feature, it's OK.
                    //       However, this means that some other DataSourceFuncotationFactory must be producing a
                    //       funcotation for this variant.
                    //       For it is decreed that all variants must have a funcotation, even if that funcotation be
                    //       empty.
                    // TODO: Actually you may want to put another IGR creation here for now...
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
     * @param allele The allele of the given variant to annotate.
     * @param gtfFeature The GTF feature on which to base annotations.
     * @return A {@link List} of {@link GencodeFuncotation}s for the given variant, allele and gtf feature.
     */
    private List<GencodeFuncotation> createFuncotations(final VariantContext variant, final Allele allele, final GencodeGtfGeneFeature gtfFeature, final ReferenceContext reference) {
        // For each applicable transcript, create an annotation.

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        final SimpleInterval variantPosition = new SimpleInterval(variant.getContig(), variant.getStart(), variant.getEnd());

        // Find our transcript:
        for (final GencodeGtfTranscriptFeature transcript : gtfFeature.getTranscripts() ) {

            boolean hasHandledVariant = false;

            final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

            if ( transcript.getGenomicPosition().overlaps(variantPosition) ) {
                // Go through our sub-features and determine what the VariantClassification is:

                // Set our allele, gene, and transcript info, since we know we have one to annotate:
                gencodeFuncotation.setAllele( allele.getBaseString() );
                gencodeFuncotation.setSymbol( gtfFeature.getGeneName() );
                gencodeFuncotation.setGene( gtfFeature.getGeneId() );
                gencodeFuncotation.setFeatureType( "Transcript" );
                gencodeFuncotation.setFeature( transcript.getTranscriptId() );
                gencodeFuncotation.setBiotype( transcript.getTranscriptType().toString() );

                // Find our Exon:
                for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {

                    // Set our Exon fields:
                    gencodeFuncotation.setFeatureType( "Exon" );
                    gencodeFuncotation.setFeature( exon.getExonId() );
                    gencodeFuncotation.setExon( exon.getExonNumber() );

                    if ( exon.getGenomicPosition().overlaps(variantPosition) ) {
                        // Find the variant in this exon:
                        if ((exon.getStartCodon() != null) && exon.getStartCodon().getGenomicPosition().overlaps(variantPosition)) {
                            // It's a start codon variant.

                            if ( variant.getType() == VariantContext.Type.SNP ) {
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_SNP );
                            }
                            else if ( variant.getType() == VariantContext.Type.MNP ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_INS );
                            }
                            else if ( variant.getType() == VariantContext.Type.INDEL ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_INS );
                            }
                            else {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.START_CODON_DEL );
                            }

                        } else if ((exon.getCds() != null) && exon.getCds().getGenomicPosition().overlaps(variantPosition)) {
                            // It's a 'normal' variant in the coding region.

                            final GencodeFuncotation.VariantClassification variantClass;

                            switch (variant.getType()) {
                                case SNP:
                                    variantClass = getVariantClassificationSnp(variant, allele, reference, transcript);
                                    break;
                                case MIXED:
                                case MNP:
                                case SYMBOLIC:
                                case INDEL:
                                    if ( isFrameshift(variant.getReference(), allele) ) {
                                        // TODO: THIS IS ALMOST CERTAINLY WRONG:
                                        if ( allele.length() < variant.getReference().length()) {
                                            variantClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL;
                                        }
                                        else if ( allele.length() > variant.getReference().length()) {
                                            variantClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS;
                                        }
                                        else {
                                            variantClass = GencodeFuncotation.VariantClassification.FRAME_SHIFT_SUB;
                                        }
                                    }
                                    else {
                                        if ( allele.length() < variant.getReference().length()) {
                                            variantClass = GencodeFuncotation.VariantClassification.IN_FRAME_DEL;
                                        }
                                        else if ( allele.length() > variant.getReference().length()) {
                                            variantClass = GencodeFuncotation.VariantClassification.IN_FRAME_INS;
                                        }
                                        else {
                                            variantClass = getVariantClassificationSnp(variant, allele, reference, transcript);
                                        }
                                    }
                                    break;
                                case NO_VARIATION:
                                default:
                                    variantClass = GencodeFuncotation.VariantClassification.SILENT;
                                    break;
                            }

                            gencodeFuncotation.setClassification( variantClass );

                            // Get the CDS position:
                            final int cdsPosition = variantPosition.getStart() - exon.getCds().getGenomicPosition().getStart();
                            gencodeFuncotation.setCdsPosition( cdsPosition );
                            gencodeFuncotation.setProteinPosition( (int)Math.ceil(cdsPosition/3.0) );

                        } else {
                            // Must be a stop_codon variant.

                            if ( variant.getType() == VariantContext.Type.SNP ) {
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_SNP );
                            }
                            else if ( variant.getType() == VariantContext.Type.MNP ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_INS );
                            }
                            else if ( variant.getType() == VariantContext.Type.INDEL ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_INS );
                            }
                            else {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                gencodeFuncotation.setClassification( GencodeFuncotation.VariantClassification.STOP_CODON_DEL );
                            }

                        }

                        gencodeFuncotations.add(gencodeFuncotation);
                        hasHandledVariant = true;
                    }
                }
                if ( hasHandledVariant ) { continue; }

                // Find our UTR:
                for ( final GencodeGtfUTRFeature utr : transcript.getUtrs() ) {
                    if ( utr.getGenomicPosition().overlaps(variantPosition) ) {
                        // We need to determine if this is a 3' or 5' UTR variation

                        // There are 2 UTRs.  One is at the front (3'), the other at the back (5'):
                        if ( utr.getGenomicStartLocation() == transcript.getStart() ) {
                            gencodeFuncotation.setClassification(GencodeFuncotation.VariantClassification.THREE_PRIME_UTR);
                        }
                        else {
                            gencodeFuncotation.setClassification(GencodeFuncotation.VariantClassification.FIVE_PRIME_UTR);
                        }

                        gencodeFuncotations.add(gencodeFuncotation);
                        hasHandledVariant = true;
                    }
                }
                if ( hasHandledVariant ) { continue; }

                // NOTE: We do not have to handle the Selenocysteines differently.
                // They will be contained in CDS regions and will be handled there.

                // Must be an intron:
                gencodeFuncotation.setClassification(GencodeFuncotation.VariantClassification.INTRON);

                gencodeFuncotations.add(gencodeFuncotation);
            }
        }

        return gencodeFuncotations;
    }

    /**
     * Creates a {@link List} of {@link GencodeFuncotation}s based on the given {@link VariantContext}.
     * These are most likely to be {@link GencodeFuncotation.VariantClassification#IGR}, but could also be
     * {@link GencodeFuncotation.VariantClassification#DE_NOVO_START_IN_FRAME} or {@link GencodeFuncotation.VariantClassification#DE_NOVO_START_OUT_FRAME}
     * @param variant The variant to annotate.
     * @param reference The reference against which to compare the given variant.
     * @return A list of IGR annotations for the given variant.
     */
    private List<GencodeFuncotation> createIgrFuncotations(final VariantContext variant, final ReferenceContext reference) {
        // for each allele, create an annotation.

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        for ( final Allele allele : variant.getAlternateAlleles() ) {
            gencodeFuncotations.add( createIgrAnnotationForAllele(allele, reference) );
        }

        return gencodeFuncotations;
    }

    /**
     * Create a {@link GencodeFuncotation} representing the intergenic region variant given by {@code allele} and the {@code reference}.
     * @param allele An {@link Allele} from which to create a {@link GencodeFuncotation}.
     * @param reference An {@link ReferenceContext} on which to base the resulting {@link GencodeFuncotation}.
     * @return A {@link GencodeFuncotation} corresponding to an IGR variant given an allele and reference.
     */
    private GencodeFuncotation createIgrAnnotationForAllele(final Allele allele, final ReferenceContext reference) {
        // Start setting up our GencodeFuncotation:
        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();
        gencodeFuncotation.setTumorSeqAllele1( allele.getBaseString() );
        gencodeFuncotation.setTumorSeqAllele2( allele.getBaseString() );

        // Determine if we're an IGR or a DE_NOVO_START:
        // TODO: EVERYTHING IS AN IGR FOR NOW - FIX THIS!
        gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.IGR);

        return gencodeFuncotation;
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
