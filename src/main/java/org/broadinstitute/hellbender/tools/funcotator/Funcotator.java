package org.broadinstitute.hellbender.tools.funcotator;

import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.variantcontext.VariantContextBuilder;
import htsjdk.variant.variantcontext.writer.VariantContextWriter;
import htsjdk.variant.vcf.*;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.BetaFeature;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.VariantProgramGroup;
import org.broadinstitute.hellbender.engine.*;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;

import java.io.File;
import java.util.*;

import htsjdk.variant.variantcontext.VariantContext.Type;

/**
 * Funcotator (FUNCtional annOTATOR) performs functional analysis on given variants
 * and reports output in a specified output file.
 *
 * This tool is the GATK analog of the Oncotator.
 *
 * Created by jonn on 8/22/17.
 */
@CommandLineProgramProperties(
        summary = "Create functional annotations on given variants cross-referenced by a given database.\n" +
                "A GATK version of the Oncotator.",
        oneLineSummary = "(Experimental) Functional Annotator",
        programGroup = VariantProgramGroup.class
)
@DocumentedFeature
@BetaFeature
public class Funcotator extends VariantWalker {

    /**
     * The name of the field inside the VCF INFO field in which to put {@link Funcotator} results.
     */
    private static final String FUNCOTATOR_VCF_FIELD_NAME = "FUNCOTATOR";

    //==================================================================================================================

    @Argument(
            shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME,
            fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME,
            doc="Output VCF File.")
    protected File outputFile;

    @Argument(
            fullName="gtfFile",
            shortName="gtf",
            doc="A GENCODE GTF file containing annotated genes."
    )
    private FeatureInput<GencodeGtfFeature> gtfVariants;

    //==================================================================================================================

    private VariantContextWriter vcfWriter;

    //==================================================================================================================

    /**
     * Determines whether the given reference and alternate alleles constitute a frameshift mutation.
     * @param reference The reference {@link Allele}.
     * @param alternate The alternate / variant {@link Allele}.
     * @return {@code true} if replacing the reference with the alternate results in a frameshift.  {@code false} otherwise.
     */
    public static boolean isFrameshift(final Allele reference, final Allele alternate) {

        // We know it's a frameshift if we have a replacement that is not of a
        // length evenly divisible by 3 because that's how many bases are read at once:
        return ((Math.abs( reference.length() - alternate.length() ) % 3) == 0);
    }

    //==================================================================================================================

    @Override
    public boolean requiresReference() {
        return true;
    }

    @Override
    public void onTraversalStart() {
        logger.info("Beginning VCF traversal");

        // Func-ify the output VCF:
        setupVCFWriter();
    }


    @Override
    public void apply(final VariantContext variant, final ReadsContext readsContext, final ReferenceContext referenceContext, final FeatureContext featureContext) {

        if ( !referenceContext.hasBackingDataSource() ) {
            throw new GATKException("No reference context for IGR variant.  Cannot annotate!");
        }

        final List<Funcotation> funcotations = new ArrayList<>();

        // If we have overlapping data with this variant, go through our known features:
        if ( featureContext.hasBackingDataSource() ) {

            // Get our known GTF features:
            final List<GencodeGtfFeature> gtfFeatures = featureContext.getValues(gtfVariants);

            for ( final Allele allele : variant.getAlternateAlleles() ) {
                for ( final GencodeGtfFeature gtfFeature : gtfFeatures ) {
                    funcotations.addAll( createFuncotations(variant, allele, (GencodeGtfGeneFeature)gtfFeature, referenceContext) );
                }
            }
        }
        else {
            // This is an IGR.
            funcotations.addAll( createIgrFuncotations(variant, referenceContext) );
        }

        // Create a new variant context builder:
        final VariantContextBuilder variantContextBuilder = new VariantContextBuilder(variant);

        final StringBuilder funcotatorAnnotationStringBuilder = new StringBuilder();

        // Get the old VCF Annotation field and append the new information to it:
        final Object existingAnnotation = variant.getAttribute(FUNCOTATOR_VCF_FIELD_NAME, null);
        if ( existingAnnotation != null) {
            funcotatorAnnotationStringBuilder.append( existingAnnotation.toString() );
            funcotatorAnnotationStringBuilder.append( ',' );
        }

        for ( final Funcotation funcotation : funcotations ) {
            funcotatorAnnotationStringBuilder.append( funcotation.serializeToVcfString() );
            funcotatorAnnotationStringBuilder.append( ',' );
        }
        // Remove trailing ',':
        funcotatorAnnotationStringBuilder.deleteCharAt( funcotatorAnnotationStringBuilder.length() - 1 );

        // Add our new annotation and render the VariantContext:
        variantContextBuilder.attribute(FUNCOTATOR_VCF_FIELD_NAME, funcotatorAnnotationStringBuilder.toString());
        vcfWriter.add( variantContextBuilder.make() );
    }

    @Override
    public Object onTraversalSuccess() {
        logger.info("VCF traversal complete!");
        return null;
    }

    @Override
    public void closeTool() {
        vcfWriter.close();
    }

    //==================================================================================================================

    /**
     * Creates and initializes {@link Funcotator#vcfWriter}.
     * After calling this method, {@link Funcotator#vcfWriter} will be ready to write records.
     */
    private void setupVCFWriter() {
        vcfWriter = createVCFWriter(outputFile);
        vcfWriter.writeHeader(createVCFHeader());
    }

    /**
     * Create a header for a VCF file.
     * @return The {@link VCFHeader} object with relevant information for {@link Funcotator}.
     */
    private static VCFHeader createVCFHeader() {
        final Set<VCFHeaderLine> headerLines = new HashSet<>();

        headerLines.add(new VCFFormatHeaderLine(FUNCOTATOR_VCF_FIELD_NAME, VCFHeaderLineCount.UNBOUNDED,
                VCFHeaderLineType.String, "Functional annotation from the Funcotator tool.")
        );

        return new VCFHeader(headerLines);
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
     * Creates a {@link List} of {@link Funcotation}s based on the given {@link VariantContext}, {@link Allele}, and {@link GencodeGtfGeneFeature}.
     * @param variant The variant to annotate.
     * @param allele The allele of the given variant to annotate.
     * @param gtfFeature The GTF feature on which to base annotations.
     * @return A {@link List} of {@link Funcotation}s for the given variant, allele and gtf feature.
     */
    private List<Funcotation> createFuncotations(final VariantContext variant, final Allele allele, final GencodeGtfGeneFeature gtfFeature, final ReferenceContext reference) {
        // For each applicable transcript, create an annotation.

        final List<Funcotation> funcotations = new ArrayList<>();

        final SimpleInterval variantPosition = new SimpleInterval(variant.getContig(), variant.getStart(), variant.getEnd());

        // Find our transcript:
        for (final GencodeGtfTranscriptFeature transcript : gtfFeature.getTranscripts() ) {

            boolean hasHandledVariant = false;

            final Funcotation funcotation = new Funcotation();

            if ( transcript.getGenomicPosition().overlaps(variantPosition) ) {
                // Go through our sub-features and determine what the VariantClassification is:

                // Set our allele, gene, and transcript info, since we know we have one to annotate:
                funcotation.setAllele( allele.getBaseString() );
                funcotation.setSymbol( gtfFeature.getGeneName() );
                funcotation.setGene( gtfFeature.getGeneId() );
                funcotation.setFeatureType( "Transcript" );
                funcotation.setFeature( transcript.getTranscriptId() );
                funcotation.setBiotype( transcript.getTranscriptType().toString() );

                // Find our Exon:
                for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {

                    // Set our Exon fields:
                    funcotation.setFeatureType( "Exon" );
                    funcotation.setFeature( exon.getExonId() );
                    funcotation.setExon( exon.getExonNumber() );

                    if ( exon.getGenomicPosition().overlaps(variantPosition) ) {
                        // Find the variant in this exon:
                        if (exon.getStartCodon().getGenomicPosition().overlaps(variantPosition)) {
                            // It's a start codon variant.

                            if ( variant.getType() == Type.SNP ) {
                                funcotation.setClassification( Funcotation.VariantClassification.START_CODON_SNP );
                            }
                            else if ( variant.getType() == Type.MNP ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                funcotation.setClassification( Funcotation.VariantClassification.START_CODON_INS );
                            }
                            else if ( variant.getType() == Type.INDEL ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                funcotation.setClassification( Funcotation.VariantClassification.START_CODON_INS );
                            }
                            else {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                funcotation.setClassification( Funcotation.VariantClassification.START_CODON_DEL );
                            }

                        } else if (exon.getCds().getGenomicPosition().overlaps(variantPosition)) {
                            // It's a 'normal' variant in the coding region.

                            final Funcotation.VariantClassification variantClass;

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
                                            variantClass = Funcotation.VariantClassification.FRAME_SHIFT_DEL;
                                        }
                                        else if ( allele.length() > variant.getReference().length()) {
                                            variantClass = Funcotation.VariantClassification.FRAME_SHIFT_INS;
                                        }
                                        else {
                                            variantClass = Funcotation.VariantClassification.FRAME_SHIFT_SUB;
                                        }
                                    }
                                    else {
                                        if ( allele.length() < variant.getReference().length()) {
                                            variantClass = Funcotation.VariantClassification.IN_FRAME_DEL;
                                        }
                                        else if ( allele.length() > variant.getReference().length()) {
                                            variantClass = Funcotation.VariantClassification.IN_FRAME_INS;
                                        }
                                        else {
                                            variantClass = getVariantClassificationSnp(variant, allele, reference, transcript);
                                        }
                                    }
                                    break;
                                case NO_VARIATION:
                                default:
                                    variantClass = Funcotation.VariantClassification.SILENT;
                                    break;
                            }

                            funcotation.setClassification( variantClass );

                            // Get the CDS position:
                            final int cdsPosition = variantPosition.getStart() - exon.getCds().getGenomicPosition().getStart();
                            funcotation.setCdsPosition( cdsPosition );
                            funcotation.setProteinPosition( (int)Math.ceil(cdsPosition/3.0) );

                        } else {
                            // Must be a stop_codon variant.

                            if ( variant.getType() == Type.SNP ) {
                                funcotation.setClassification( Funcotation.VariantClassification.STOP_CODON_SNP );
                            }
                            else if ( variant.getType() == Type.MNP ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                funcotation.setClassification( Funcotation.VariantClassification.STOP_CODON_INS );
                            }
                            else if ( variant.getType() == Type.INDEL ) {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                funcotation.setClassification( Funcotation.VariantClassification.STOP_CODON_INS );
                            }
                            else {
                                // TODO: FIX THIS IT'S WRONG - NEED TO DETERMINE INSERTION OR DELETION
                                funcotation.setClassification( Funcotation.VariantClassification.STOP_CODON_DEL );
                            }

                        }

                        funcotations.add(funcotation);
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
                            funcotation.setClassification(Funcotation.VariantClassification.THREE_PRIME_UTR);
                        }
                        else {
                            funcotation.setClassification(Funcotation.VariantClassification.FIVE_PRIME_UTR);
                        }

                        funcotations.add(funcotation);
                        hasHandledVariant = true;
                    }
                }
                if ( hasHandledVariant ) { continue; }

                // NOTE: We do not have to handle the Selenocysteines differently.
                // They will be contained in CDS regions and will be handled there.

                // Must be an intron:
                funcotation.setClassification(Funcotation.VariantClassification.INTRON);

                funcotations.add(funcotation);
            }
        }

        return funcotations;
    }

    private Funcotation.VariantClassification getVariantClassificationSnp(final VariantContext variant,
                                                                          final Allele allele,
                                                                          final ReferenceContext reference,
                                                                          final GencodeGtfTranscriptFeature transcript) {
        final Funcotation.VariantClassification variantClass;

        // the reference is aligned exactly to the variant.
        // We need to line up the reference and the variant to the transcript.
        final int variantStartOffset = calculateOffsetToTranscriptAlignment(variant, transcript);

        // TODO: This is insufficient because of splice sites:
        reference.setWindow(variant.getStart() - variantStartOffset, 3);
        final byte[] rawBases = reference.getBases();
        final byte[] bases = new byte[] { rawBases[0], rawBases[1], rawBases[2] };

        final byte[] variantBases = bases.clone();
        variantBases[-variantStartOffset] = allele.getBases()[0];

        final AminoAcid referenceAminoAcid = AminoAcidUtils.getEukaryoticAminoAcidByCodon( new String(bases) );
        final AminoAcid variantAminoAcid = AminoAcidUtils.getEukaryoticAminoAcidByCodon( new String(variantBases) );

        if ( variantAminoAcid == null ) {
            variantClass = Funcotation.VariantClassification.NONSENSE;
        }
        else {
            if ( referenceAminoAcid == variantAminoAcid ) {
                variantClass = Funcotation.VariantClassification.SILENT;
            }
            else {
                variantClass = Funcotation.VariantClassification.MISSENSE;
            }
        }

        return variantClass;
    }

    /**
     * Creates a {@link List} of {@link Funcotation}s based on the given {@link VariantContext}.
     * These are most likely to be {@link Funcotation.VariantClassification#IGR}, but could also be
     * {@link Funcotation.VariantClassification#DE_NOVO_START_IN_FRAME} or {@link Funcotation.VariantClassification#DE_NOVO_START_OUT_FRAME}
     * @param variant The variant to annotate.
     * @param reference The reference against which to compare the given variant.
     * @return A list of IGR annotations for the given variant.
     */
    private List<Funcotation> createIgrFuncotations(final VariantContext variant, final ReferenceContext reference) {
        // for each allele, create an annotation.

        final List<Funcotation> funcotations = new ArrayList<>();

        for ( final Allele allele : variant.getAlternateAlleles() ) {

            // Start setting up our Funcotation:
            final Funcotation funcotation = new Funcotation();
            funcotation.setAllele( allele.getBaseString() );

            // Determine if we're an IGR or a DE_NOVO_START:
            // TODO: EVERYTHING IS AN IGR FOR NOW - FIX THIS!
            funcotation.setClassification(Funcotation.VariantClassification.IGR);

            // Add the funcotation to the list:
            funcotations.add( funcotation );
        }

        return funcotations;
    }
}
