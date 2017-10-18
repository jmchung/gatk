package org.broadinstitute.hellbender.tools.copynumber;

import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.tools.copynumber.annotation.AnnotatedInterval;
import org.broadinstitute.hellbender.tools.copynumber.annotation.AnnotatedIntervalCollection;
import org.broadinstitute.hellbender.tools.copynumber.annotation.AnnotationSet;
import org.broadinstitute.hellbender.tools.copynumber.formats.collections.LocatableCollection;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.test.ArgumentsBuilder;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.util.Arrays;

/**
 * Integration tests for {@link AnnotateIntervals}.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class AnnotateIntervalsIntegrationTest extends CommandLineProgramTest {
    private static final String TEST_SUB_DIR = toolsTestDir + "copynumber/";
    private static final File INTERVAL_LIST_FILE = new File(TEST_SUB_DIR, "preprocess-intervals-test.interval_list");
    private static final File REFERENCE_FILE = new File(b37_reference_20_21);

    /**
     * Test that intervals are sorted according to {@link LocatableCollection#LEXICOGRAPHICAL_ORDER_COMPARATOR}.
     * GC content truth was taken from AnnotateTargets (a previous version of the tool).
     */
    @Test
    public void test() {
        final File outputFile = createTempFile("annotate-intervals-test", ".tsv");
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder()
                .addReference(REFERENCE_FILE)
                .addArgument("L",  INTERVAL_LIST_FILE.getAbsolutePath())
                .addOutput(outputFile);
        runCommandLine(argsBuilder);
        final AnnotatedIntervalCollection result = new AnnotatedIntervalCollection(outputFile);

        final AnnotatedIntervalCollection expected = new AnnotatedIntervalCollection(Arrays.asList(
                new AnnotatedInterval(new SimpleInterval("20", 100,	200), new AnnotationSet(Double.NaN)),
                new AnnotatedInterval(new SimpleInterval("20", 400,	600), new AnnotationSet(Double.NaN)),
                new AnnotatedInterval(new SimpleInterval("20", 10_000,	11_000), new AnnotationSet(Double.NaN)),
                new AnnotatedInterval(new SimpleInterval("21", 1,	100), new AnnotationSet(Double.NaN)),
                new AnnotatedInterval(new SimpleInterval("21", 20_000,	22_000), new AnnotationSet(Double.NaN))));

        Assert.assertEquals(result, expected);
        Assert.assertNotSame(result, expected);
    }
}