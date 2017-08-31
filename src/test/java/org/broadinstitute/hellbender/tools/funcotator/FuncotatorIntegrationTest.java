package org.broadinstitute.hellbender.tools.funcotator;

import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jonn on 8/29/17.
 */
public class FuncotatorIntegrationTest extends CommandLineProgramTest {

//    ./gatk-launch Funcotator \
//            -gtf /Users/jonn/Downloads/gencode.v19.chr_patch_hapl_scaff.annotation.fixed.truncated.gtf \
//            -O   test_funcotator_out.vcf \
//            -R   /Users/jonn/Development/references/GRCh37.p13.genome.fasta \
//            -V   /Users/jonn/Development/gatk/src/test/resources/org/broadinstitute/hellbender/tools/funcotator/singleSnpTest_chr2_hg19.vcf \
//            -- \
//            --javaOptions '-DGATK_STACKTRACE_ON_USER_EXCEPTION=true'

    @Test
    public void testRun() throws IOException {
        final File outputFile = File.createTempFile("funcotator_tmp_out", ".vcf");
        final List<String> arguments = new ArrayList<>();

        arguments.add("-" + Funcotator.GTF_FILE_SHORT_NAME);
        arguments.add("/Users/jonn/Downloads/gencode.v19.chr_patch_hapl_scaff.annotation.fixed.truncated.gtf");
        arguments.add("-" + StandardArgumentDefinitions.REFERENCE_SHORT_NAME);
        arguments.add("/Users/jonn/Development/references/GRCh37.p13.genome.fasta");
        arguments.add("-" + StandardArgumentDefinitions.VARIANT_SHORT_NAME);
        arguments.add("/Users/jonn/Development/gatk/src/test/resources/org/broadinstitute/hellbender/tools/funcotator/singleSnpTest_chr2_hg19.vcf");
        arguments.add("-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME);
        arguments.add(outputFile.getAbsolutePath());
        runCommandLine(arguments);
    }

}
