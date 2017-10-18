package org.broadinstitute.hellbender.tools.copynumber.temporary;

import org.broadinstitute.hdf5.HDF5File;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HDF5ReadCountCollectionUnitTest extends BaseTest {

    @Test
    public void basicTest() {
        final File outputFile = createTempFile("HDF5ReadCountCollection", ".hdf5");
        final String sampleName = "SAMPLE1";
        final List<SimpleInterval> intervals = new ArrayList<>();
        intervals.add(new SimpleInterval("1", 1000, 2000));
        intervals.add(new SimpleInterval("1", 5000, 6000));
        final double[][] readCounts = {{2, 10}};

        // The output file already exists at this point, since it is a temp file.
        HDF5ReadCountCollection.write(outputFile, sampleName, intervals, readCounts);

        final HDF5ReadCountCollection rcc = new HDF5ReadCountCollection(new HDF5File(outputFile));

        Assert.assertEquals(rcc.getSampleName(), sampleName);

        Assert.assertEquals(rcc.getIntervals(), intervals);
        Assert.assertFalse(rcc.getIntervals() == intervals);

        Assert.assertEquals(rcc.getReadCounts().getRowDimension(), 1);
        Assert.assertTrue(Arrays.deepEquals(rcc.getReadCounts().getData(), readCounts));
        Assert.assertFalse(rcc.getReadCounts().getData() == readCounts);
    }
}
