package org.broadinstitute.hellbender.tools.copynumber.allelic.model;

import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SampleMetadata;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SimpleSampleMetadata;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Test the initialization on the allele fraction model
 * @author David Benjamin
 */
public final class AlleleFractionInitializerUnitTest {
    @Test
    public void testInitialize() {
        final SampleMetadata sampleMetadata = new SimpleSampleMetadata("test");
        final double averageHetsPerSegment = 50;
        final int numSegments = 100;
        final int averageDepth = 50;

        final double biasMean = 1.1;
        final double biasVariance = 0.01;
        final double outlierProbability = 0.02;

        final double minorFractionTolerance = 0.02;
        final double biasMeanTolerance = 0.02;
        final double biasVarianceTolerance = 0.01;
        final double outlierProbabilityTolerance = 0.01;

        final AlleleFractionSimulatedData simulatedData = new AlleleFractionSimulatedData(
                sampleMetadata, averageHetsPerSegment, numSegments, averageDepth, biasMean, biasVariance, outlierProbability);

        final AlleleFractionSegmentedData data = simulatedData.getData();
        final AlleleFractionState initializedState = new AlleleFractionInitializer(data).getInitializedState();

        final AlleleFractionSimulatedData.AlleleFractionStateError error = simulatedData.error(initializedState);
        Assert.assertEquals(error.averageMinorFractionError, 0, minorFractionTolerance);
        Assert.assertEquals(error.biasMeanError, 0, biasMeanTolerance);
        Assert.assertEquals(error.biasVarianceError, 0, biasVarianceTolerance);
        Assert.assertEquals(error.outlierProbabilityError, 0, outlierProbabilityTolerance);
    }
}