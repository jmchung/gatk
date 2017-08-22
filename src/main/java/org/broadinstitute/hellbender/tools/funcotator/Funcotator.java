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
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.exome.orientationbiasvariantfilter.OrientationBiasFilterConstants;
import org.broadinstitute.hellbender.tools.exome.orientationbiasvariantfilter.OrientationBiasFilterer;
import org.broadinstitute.hellbender.tools.walkers.annotator.VariantAnnotatorEngine;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.GencodeGtfFeature;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.GencodeGtfGeneFeature;
import org.broadinstitute.hellbender.utils.genotyper.IndexedSampleList;
import org.broadinstitute.hellbender.utils.genotyper.SampleList;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.*;

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
            doc="Output VCF File",
            shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME,
            fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME)
    protected File outputFile;

    @Argument(
            fullName="gcfVariants",
            shortName="gcf",
            doc="A set of GCF variants."
    )
    private FeatureInput<GencodeGtfFeature> gcfVariants;

    //==================================================================================================================

    private VariantContextWriter vcfWriter;

    //==================================================================================================================

    @Override
    public void onTraversalStart() {
        logger.info("Beginning VCF traversal");

        // Func-ify the output VCF:
        setupVCFWriter();
    }


    @Override
    public void apply(final VariantContext variant, final ReadsContext readsContext, final ReferenceContext referenceContext, final FeatureContext featureContext) {

        final List<Funcotation> funcotations = new ArrayList<>();

        // If we have overlapping data with this variant, go through our known features:
        if ( featureContext.hasBackingDataSource() ) {

            // Get our known GTF features:
            final List<GencodeGtfFeature> gtfFeatures = featureContext.getValues(gcfVariants);

            for ( final Allele allele : variant.getAlleles() ) {
                for ( final GencodeGtfFeature gtfFeature : gtfFeatures ) {
                    funcotations.addAll( createFuncotations(variant, allele, (GencodeGtfGeneFeature)gtfFeature) );
                }
            }
        }
        else {
            // This is an IGR.
            funcotations.addAll( createIgrFuncotations(variant) );
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
            funcotatorAnnotationStringBuilder.append( funcotation.serializeToString() );
        }

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

    private static VCFHeader createVCFHeader() {
        final Set<VCFHeaderLine> headerLines = new HashSet<>();

        headerLines.add(new VCFFormatHeaderLine(FUNCOTATOR_VCF_FIELD_NAME, VCFHeaderLineCount.UNBOUNDED,
                VCFHeaderLineType.String, "Functional annotation from the Funcotator tool.")
        );

        return new VCFHeader(headerLines);
    }

    /**
     * Creates a {@link List} of {@link Funcotation}s based on the given {@link VariantContext}, {@link Allele}, and {@link GencodeGtfGeneFeature}.
     * @param variant The variant to annotate.
     * @param allele The allele of the given variant to annotate.
     * @param gtfFeature The GTF feature on which to base annotations.
     * @return
     */
    private List<Funcotation> createFuncotations(VariantContext variant, Allele allele, GencodeGtfGeneFeature gtfFeature) {
        // For each applicable transcript, create an annotation.
    }

    /**
     * Creates a {@link List} of {@link Funcotation}s based on the given {@link VariantContext}.
     * These are most likely to be {@link Funcotator.VariantClassification#IGR}, but could also be
     * {@link Funcotator.VariantClassification#DE_NOVO_START_IN_FRAME} or {@link Funcotator.VariantClassification#DE_NOVO_START_OUT_FRAME}
     * @param variant
     * @return
     */
    private List<Funcotation> createIgrFuncotations(VariantContext variant) {
        // for each allele, create an annotation.
    }

    //==================================================================================================================

    /**
     * Represents the type of variant found.
     */
    enum VariantClassification {
        INTRON(10),
        FIVE_PRIME_UTR(6),
        THREE_PRIME_UTR(6),
        IGR(20),
        FIVE_PRIME_PRIME_FLANK(15),
        THREE_PRIME_PRIME_FLANK(15),
        MISSENSE(1),
        NONSENSE(0),
        NONSTOP(0),
        SILENT(5),
        SPLICE_SITE(4),
        IN_FRAME_DEL(1),
        IN_FRAME_INS(1),
        FRAME_SHIFT_INS(2),
        FRAME_SHIFT_DEL(2),
        FRAME_SHIFT_SUB(2),
        START_CODON_SNP(3),
        START_CODON_INS(3),
        START_CODON_DEL(3),
        STOP_CODON_INS(3),
        STOP_CODON_DEL(3),
        STOP_CODON_SNP(3),
        DE_NOVO_START_IN_FRAME(1),
        DE_NOVO_START_OUT_FRAME(0),
        RNA(4),
        LINCRNA(4);

        final private int relativeSeverity;

        VariantClassification(final int sev) {
            relativeSeverity = sev;
        }


    }
}
