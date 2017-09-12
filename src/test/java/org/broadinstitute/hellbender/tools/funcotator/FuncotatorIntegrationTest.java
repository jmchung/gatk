package org.broadinstitute.hellbender.tools.funcotator;

import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * An integration test for the {@link Funcotator} tool.
 * Created by jonn on 8/29/17.
 */
public class FuncotatorIntegrationTest extends CommandLineProgramTest {

    private final String GENCODE_TRANSCRIPT_FASTA_FILE_NAME = "/Users/jonn/Development/references/gencode.v19.pc_transcripts.fasta";

    private final String GTF_FILE_NAME = "/Users/jonn/Downloads/gencode.v19.chr_patch_hapl_scaff.annotation.fixed.truncated.gtf";
    private final String HG19_REFERENCE_FILE_NAME =  "/Users/jonn/Development/references/GRCh37.p13.genome.fasta";

    private final String VARIANT_FILE_HG19_CHR1 = "/Users/jonn/Development/gatk/src/test/resources/org/broadinstitute/hellbender/tools/funcotator/singleSnpTest_chr1_hg19.vcf";
    private final String VARIANT_FILE_HG19_CHR2 = "/Users/jonn/Development/gatk/src/test/resources/org/broadinstitute/hellbender/tools/funcotator/singleSnpTest_chr2_hg19.vcf";

    //==================================================================================================================

    @DataProvider
    Object[][] provideDataForIntegrationTest() {
        return new Object[][] {
//                {GTF_FILE_NAME, HG19_REFERENCE_FILE_NAME, GENCODE_TRANSCRIPT_FASTA_FILE_NAME, VARIANT_FILE_HG19_CHR1},
                {GTF_FILE_NAME, HG19_REFERENCE_FILE_NAME, GENCODE_TRANSCRIPT_FASTA_FILE_NAME, VARIANT_FILE_HG19_CHR2},
        };
    }

    //==================================================================================================================

    // TODO: REFACTOR ALL THIS:

    @Test(dataProvider = "provideDataForIntegrationTest")
    public void testRun(final String gtfFileName, final String referenceFileName, final String fastaFileName, final String variantFileName) throws IOException {
//        final File outputFile = BaseTest.createTempFile("funcotator_tmp_out", ".vcf");
        final File outputFile = new File("funcotator_tmp_out" + ".vcf");
        final List<String> arguments = new ArrayList<>();

        arguments.add("-" + Funcotator.GTF_FILE_ARG_SHORT_NAME);
        arguments.add(gtfFileName);
        arguments.add("-" + StandardArgumentDefinitions.REFERENCE_SHORT_NAME);
        arguments.add(referenceFileName);
        arguments.add("-" + Funcotator.GENCODE_FASTA_ARG_NAME);
        arguments.add(fastaFileName);
        arguments.add("-" + StandardArgumentDefinitions.VARIANT_SHORT_NAME);
        arguments.add(variantFileName);
        arguments.add("-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME);
        arguments.add(outputFile.getAbsolutePath());

        runCommandLine(arguments);
    }
}
