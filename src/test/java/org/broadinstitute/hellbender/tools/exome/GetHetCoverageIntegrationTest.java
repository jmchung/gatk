package org.broadinstitute.hellbender.tools.exome;

import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.SamReader;
import htsjdk.samtools.SamReaderFactory;
import htsjdk.samtools.ValidationStringency;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.cmdline.ExomeStandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;

/**
 * Integration test for {@link GetHetCoverage}.  Uses BAM and SNP files generated from hg19mini using wgsim.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class GetHetCoverageIntegrationTest extends CommandLineProgramTest {
    private static final String TEST_SUB_DIR = publicTestDir + "org/broadinstitute/hellbender/tools/exome";

    private static final File NORMAL_BAM_FILE = new File(TEST_SUB_DIR, "normal.sorted.bam");
    private static final File TUMOR_BAM_FILE = new File(TEST_SUB_DIR, "tumor.sorted.bam");
    private static final File NON_STRICT_BAM_FILE = new File(TEST_SUB_DIR, "simple_overhang.sam");
    private static final File SNP_FILE = new File(TEST_SUB_DIR, "common_SNP.interval_list");
    private static final File REF_FILE = new File(hg19MiniReference);

    private static SAMFileHeader normalHeader;
    private static SAMFileHeader tumorHeader;

    @BeforeClass
    public void initHeaders() throws IOException {
        try (final SamReader normalBamReader = SamReaderFactory.makeDefault().open(NORMAL_BAM_FILE);
             final SamReader tumorBamReader = SamReaderFactory.makeDefault().open(TUMOR_BAM_FILE)) {
            normalHeader = normalBamReader.getFileHeader();
            tumorHeader = tumorBamReader.getFileHeader();
        }
    }

    @Test
    public void testGetHetCoverage() {
        final File normalOutputFile = createTempFile("normal-test", ".txt");
        final File tumorOutputFile = createTempFile("tumor-test", ".txt");

        final String[] arguments = {
                "-" + ExomeStandardArgumentDefinitions.NORMAL_BAM_FILE_SHORT_NAME, NORMAL_BAM_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.TUMOR_BAM_FILE_SHORT_NAME, TUMOR_BAM_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.SNP_FILE_SHORT_NAME, SNP_FILE.getAbsolutePath(),
                "-" + StandardArgumentDefinitions.REFERENCE_SHORT_NAME, REF_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.NORMAL_ALLELIC_COUNTS_FILE_SHORT_NAME, normalOutputFile.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.TUMOR_ALLELIC_COUNTS_FILE_SHORT_NAME, tumorOutputFile.getAbsolutePath()
        };
        runCommandLine(arguments);

        final Pulldown normalOutputPulldownCLP = new Pulldown(normalOutputFile, normalHeader);
        final Pulldown tumorOutputPulldownCLP = new Pulldown(tumorOutputFile, tumorHeader);

        Pulldown normalHetPulldown = new Pulldown(normalHeader);
        normalHetPulldown.add(new SimpleInterval("1", 11522, 11522), 7, 4);
        normalHetPulldown.add(new SimpleInterval("1", 12098, 12098), 8, 6);
        normalHetPulldown.add(new SimpleInterval("1", 14630, 14630), 9, 8);
        normalHetPulldown.add(new SimpleInterval("2", 14689, 14689), 6, 9);
        normalHetPulldown.add(new SimpleInterval("2", 14982, 14982), 6, 5);

        Pulldown tumorHetPulldown = new Pulldown(tumorHeader);
        tumorHetPulldown.add(new SimpleInterval("1", 11522, 11522), 7, 4);
        tumorHetPulldown.add(new SimpleInterval("1", 12098, 12098), 8, 6);
        tumorHetPulldown.add(new SimpleInterval("1", 14630, 14630), 9, 8);
        tumorHetPulldown.add(new SimpleInterval("2", 14689, 14689), 6, 9);
        tumorHetPulldown.add(new SimpleInterval("2", 14982, 14982), 6, 5);

        Assert.assertEquals(normalHetPulldown, normalOutputPulldownCLP);
        Assert.assertEquals(tumorHetPulldown, tumorOutputPulldownCLP);
    }

    //Regression test for https://github.com/broadinstitute/gatk-protected/issues/373
    @Test(expectedExceptions = UserException.class)
    public void testNonStrictBAM() {
        final File normalOutputFile = createTempFile("normal-test",".txt");
        final File tumorOutputFile = createTempFile("tumor-test",".txt");

        final String[] arguments = {
                "-" + ExomeStandardArgumentDefinitions.NORMAL_BAM_FILE_SHORT_NAME, NON_STRICT_BAM_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.TUMOR_BAM_FILE_SHORT_NAME, TUMOR_BAM_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.SNP_FILE_SHORT_NAME, SNP_FILE.getAbsolutePath(),
                "-" + StandardArgumentDefinitions.REFERENCE_SHORT_NAME, REF_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.NORMAL_ALLELIC_COUNTS_FILE_SHORT_NAME, normalOutputFile.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.TUMOR_ALLELIC_COUNTS_FILE_SHORT_NAME, tumorOutputFile.getAbsolutePath(),
                "--VALIDATION_STRINGENCY", ValidationStringency.STRICT.toString()
        };
        runCommandLine(arguments);
        //should catch SAMFormatException and throw new UserException with --VALIDATION_STRINGENCY STRICT
    }

    //Regression test for https://github.com/broadinstitute/gatk-protected/issues/373
    @Test
    public void testNonStrictBAMWithSilentValidationStringency() {
        final File normalOutputFile = createTempFile("normal-test",".txt");
        final File tumorOutputFile = createTempFile("tumor-test",".txt");

        final String[] arguments = {
                "-" + ExomeStandardArgumentDefinitions.NORMAL_BAM_FILE_SHORT_NAME, NON_STRICT_BAM_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.TUMOR_BAM_FILE_SHORT_NAME, TUMOR_BAM_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.SNP_FILE_SHORT_NAME, SNP_FILE.getAbsolutePath(),
                "-" + StandardArgumentDefinitions.REFERENCE_SHORT_NAME, REF_FILE.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.NORMAL_ALLELIC_COUNTS_FILE_SHORT_NAME, normalOutputFile.getAbsolutePath(),
                "-" + ExomeStandardArgumentDefinitions.TUMOR_ALLELIC_COUNTS_FILE_SHORT_NAME, tumorOutputFile.getAbsolutePath(),
        };
        runCommandLine(arguments);
        //should complete successfully with default --VALIDATION_STRINGENCY SILENT
    }
}