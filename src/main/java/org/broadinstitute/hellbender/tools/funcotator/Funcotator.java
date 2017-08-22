package org.broadinstitute.hellbender.tools.funcotator;

import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.barclay.argparser.BetaFeature;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.programgroups.VariantProgramGroup;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.ReadsContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.VariantWalker;

/**
 * Funcotator (FUNCtional annOTATOR) performs functional analysis on given variants
 * and reports output in a specified output file.
 *
 * This tool is the GATK analog of the Oncotator.
 *
 * Created by jonn on 8/22/17.
 */
//@CommandLineProgramProperties(
//        summary = "Filter Mutect2 somatic variant calls using the Orientation Bias Filter.\n" +
//                "Used for the OxoG (G/T) and Deamination (FFPE) (C/T) artifacts that get introduced into our SNV calling.\n" +
//                "\n" +
//                "Notes:  All variants are held in RAM.\n This tool will only catch artifacts in diploid organisms.\n" +
//                " - Triallelic sites may not be considered for filtering -- the behavior of this tool is undefined for triallelic sites.\n" +
//                " - If you do not wish to filter, you must skip the filter entirely.  You cannot do a dummy/non-filter operation with this tool.\n" +
//                " - Sites-only VCF files are NOT supported.  At least one sample must be present in the VCF file for any filtering to be done.  If no samples are present in the VCF file, no filtering annotation is actually performed.\n" +
//                " - This tool was tested only with output for GATK4 Mutect and makes assumptions about the existence of fields produced by GATK4 Mutect.\n" +
//                " - ALT_F1R2 and ALT_F2R1 tags must be present in all SNP variants.\n" +
//                " - Do NOT specify artifact modes that are reverse complements of each other.  Behavior of this tool is undefined when that happens.  For example, do not specify C/A and G/T in the same run.\n" +
//                " - Any variants that are filtered in the input file are not considered for filtering here, nor are these variants used in deciding cutoff.\n" +
//                " - The Orientation Bias puts a filter tag in both the genotype (FORMAT) and variant (FILTER) fields.\n" +
//                " - In multiallelic sites, only the first alternate allele is used for filtering.\n" +
//                " - This filter should be applied last in any M2 toolchain.\n" +
//                " - Common artifacts:\n G/T (OxoG)\n C/T (deamination) ",
//        oneLineSummary = "(Experimental) Filter Mutect2 somatic variant calls using orientation bias",
//        programGroup = VariantProgramGroup.class
//)
//@DocumentedFeature
@BetaFeature
public class Funcotator extends VariantWalker {
    @Override
    public void apply(VariantContext variant, ReadsContext readsContext, ReferenceContext referenceContext, FeatureContext featureContext) {

//        for each mutation
//              for each datasource compatible with your reference
//                  annotate

    }
}
