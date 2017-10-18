package org.broadinstitute.hellbender.tools.copynumber.coverage.model;

import org.broadinstitute.hellbender.utils.test.BaseTest;

/**
 * Unit tests for {@link CopyRatioModeller}.
 * <p>
 *     Test data consisting of copy-ratio and segment files for 100 segments with 100 copy-ratio intervals each
 *     was generated using a python script.
 *     The global parameters determining the variance and the outlier probability were set to 1. and 0.05,
 *     respectively.  The segment-level mean log2 copy ratios were drawn from Uniform(-5, 5),
 *     while outlier points were drawn from Uniform(-10, 10).
 * </p>
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class CopyRatioModellerUnitTest extends BaseTest {
//    private static final File TEST_SUB_DIR = new File(toolsTestDir, "copynumber/coverage/model");
//
//    private static final File COPY_RATIOS_FILE = new File(TEST_SUB_DIR, "copy-ratio-modeller-copy-ratios.tsv");
//    private static final File COPY_RATIO_SEGMENTS_TRUTH_FILE = new File(TEST_SUB_DIR, "copy-ratio-modeller-copy-ratio-segments-truth.seg");
//    private static final File OUTLIER_INDICATORS_TRUTH_FILE = new File(TEST_SUB_DIR, "copy-ratio-modeller-outlier-indicators-truth.txt");
//
//    private static final double CREDIBLE_INTERVAL_ALPHA = 0.32;
//
//    private static final double VARIANCE_TRUTH = 1.;
//    private static final double OUTLIER_PROBABILITY_TRUTH = 0.05;
//
//    //truths for the posterior standard deviations are based on the standard deviations of the appropriate analytic
//    //posteriors, scaled appropriately for the total number of copy ratios or the average number of copy ratios per segment
//    private static final double MEAN_POSTERIOR_STANDARD_DEVIATION_MEAN_TRUTH = 0.1;                 //Gaussian with 100 points for each mean and unit variance gives 1 / sqrt(100)
//    private static final double VARIANCE_POSTERIOR_STANDARD_DEVIATION_TRUTH = 0.014;                //inverse chi-squared with 100 DOF and variance = 1
//    private static final double OUTLIER_PROBABILITY_POSTERIOR_STANDARD_DEVIATION_TRUTH = 0.0022;    //Beta for alpha ~ 516 (true # of outliers + prior alpha - 1),
//                                                                                                    //         beta ~ 9580 (true # of non-outliers + prior beta - 1)
//
//    //test specifications
//    private static final double MULTIPLES_OF_SD_THRESHOLD = 1.5;
//    private static final double RELATIVE_ERROR_THRESHOLD = 0.15;
//    private static final double FRACTION_OF_OUTLIER_INDICATORS_CORRECT_THRESHOLD = 0.98;
//    private static final int DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_1_SIGMA = 10;
//    private static final int DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_2_SIGMA = 5;
//    private static final int DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_3_SIGMA = 2;
//
//    private static final int NUM_SAMPLES = 500;
//    private static final int NUM_BURN_IN = 250;
//
//    //Calculates relative error between x and xTrue, with respect to xTrue; used for checking statistics of
//    //posterior samples below.
//    private static double relativeError(final double x, final double xTrue) {
//        return Math.abs((x - xTrue) / xTrue);
//    }
//
//    /**
//     * Tests Bayesian inference of the copy-ratio model via MCMC.
//     * <p>
//     *     Recovery of input values for the variance and outlier-probability global parameters is checked.
//     *     In particular, the true input value of the variance must fall within
//     *     {@link CopyRatioModellerUnitTest#MULTIPLES_OF_SD_THRESHOLD}
//     *     standard deviations of the posterior mean and the standard deviation of the posterior must agree
//     *     with the analytic value to within a relative error of
//     *     {@link CopyRatioModellerUnitTest#RELATIVE_ERROR_THRESHOLD} for 250 samples
//     *     (after 250 burn-in samples have been discarded).  Similar criteria are applied
//     *     to the recovery of the true input value for the outlier probability.
//     * </p>
//     * <p>
//     *     Furthermore, the number of truth values for the segment-level mean log2 copy ratios falling outside
//     *     confidence intervals of 1-sigma, 2-sigma, and 3-sigma given by the posteriors in each segment should
//     *     be roughly consistent with a normal distribution (i.e., ~32, ~5, and ~0, respectively;
//     *     we allow for errors of
//     *     {@link CopyRatioModellerUnitTest#DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_1_SIGMA},
//     *     {@link CopyRatioModellerUnitTest#DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_2_SIGMA}, and
//     *     {@link CopyRatioModellerUnitTest#DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_3_SIGMA}, respectively).
//     *     The mean of the standard deviations of the posteriors for the segment-level means should also be
//     *     recovered to within a relative error of {@link CopyRatioModellerUnitTest#RELATIVE_ERROR_THRESHOLD}.
//     * </p>
//     * <p>
//     *     Finally, the recovered values for the latent outlier-indicator parameters should agree with those used to
//     *     generate the data.  For each indicator, the recovered value (i.e., outlier or non-outlier) is taken to be
//     *     that given by the majority of posterior samples.  We require that at least
//     *     {@link CopyRatioModellerUnitTest#FRACTION_OF_OUTLIER_INDICATORS_CORRECT_THRESHOLD}
//     *     of the 10000 indicators are recovered correctly.
//     * </p>
//     * <p>
//     *     With these specifications, this unit test is not overly brittle (i.e., it should pass for a large majority
//     *     of randomly generated data sets), but it is still brittle enough to check for correctness of the sampling
//     *     (for example, specifying a sufficiently incorrect likelihood will cause the test to fail).
//     * </p>
//     */
//    @Test
//    public void testRunMCMCOnCopyRatioSegmentedGenome() throws IOException {
//        final JavaSparkContext ctx = SparkContextFactory.getTestSparkContext();
//        LoggingUtils.setLoggingLevel(Log.LogLevel.INFO);
//
//        //load data
//        final CopyRatioCollection copyRatios = new CopyRatioCollection(COPY_RATIOS_FILE);
//        final CopyRatioSegmentCollection segmentsTruth = new CopyRatioSegmentCollection(COPY_RATIO_SEGMENTS_TRUTH_FILE);
//        final CopyRatioSegmentedData data = new CopyRatioSegmentedData(copyRatios, segmentsTruth.getIntervals());
//
//        //run MCMC
//        final CopyRatioModeller modeller = new CopyRatioModeller(data);
//        modeller.fitMCMC(NUM_SAMPLES, NUM_BURN_IN);
//
//        //check statistics of global-parameter posterior samples (i.e., posterior mode and standard deviation)
//        final Map<CopyRatioParameter, PosteriorSummary> globalParameterPosteriorSummaries =
//                modeller.getGlobalParameterDeciles(CREDIBLE_INTERVAL_ALPHA, ctx);
//
//        final PosteriorSummary variancePosteriorSummary = globalParameterPosteriorSummaries.get(CopyRatioParameter.VARIANCE);
//        final double variancePosteriorCenter = variancePosteriorSummary.getCenter();
//        final double variancePosteriorStandardDeviation = (variancePosteriorSummary.getUpper() - variancePosteriorSummary.getLower()) / 2;
//        Assert.assertEquals(Math.abs(variancePosteriorCenter - VARIANCE_TRUTH),
//                0., MULTIPLES_OF_SD_THRESHOLD * VARIANCE_POSTERIOR_STANDARD_DEVIATION_TRUTH);
//        Assert.assertEquals(relativeError(variancePosteriorStandardDeviation, VARIANCE_POSTERIOR_STANDARD_DEVIATION_TRUTH),
//                0., RELATIVE_ERROR_THRESHOLD);
//
//        final PosteriorSummary outlierProbabilityPosteriorSummary = globalParameterPosteriorSummaries.get(CopyRatioParameter.OUTLIER_PROBABILITY);
//        final double outlierProbabilityPosteriorCenter = outlierProbabilityPosteriorSummary.getCenter();
//        final double outlierProbabilityPosteriorStandardDeviation = (outlierProbabilityPosteriorSummary.getUpper() - outlierProbabilityPosteriorSummary.getLower()) / 2;
//        Assert.assertEquals(Math.abs(outlierProbabilityPosteriorCenter - OUTLIER_PROBABILITY_TRUTH),
//                0., MULTIPLES_OF_SD_THRESHOLD * OUTLIER_PROBABILITY_POSTERIOR_STANDARD_DEVIATION_TRUTH);
//        Assert.assertEquals(relativeError(outlierProbabilityPosteriorStandardDeviation,
//                OUTLIER_PROBABILITY_POSTERIOR_STANDARD_DEVIATION_TRUTH), 0., RELATIVE_ERROR_THRESHOLD);
//
//        //check statistics of segment-mean posterior samples (i.e., posterior means and standard deviations)
//        final List<Double> meansTruth = new XReadLines(COPY_RATIO_SEGMENTS_TRUTH_FILE)
//                .readLines().stream().map(Double::parseDouble).collect(Collectors.toList());
//        int numMeansOutsideOneSigma = 0;
//        int numMeansOutsideTwoSigma = 0;
//        int numMeansOutsideThreeSigma = 0;
//        final int numSegments = meansTruth.size();
//        //segment-mean posteriors are expected to be Gaussian, so PosteriorSummary for
//        // {@link CopyRatioModellerUnitTest#CREDIBLE_INTERVAL_ALPHA}=0.32 is
//        //(posterior mean, posterior mean - posterior standard devation, posterior mean + posterior standard deviation)
//        final List<PosteriorSummary> meanPosteriorSummaries =
//                modeller.getSegmentMeansPosteriorSummaries(CREDIBLE_INTERVAL_ALPHA, ctx);
//        final double[] meanPosteriorStandardDeviations = new double[numSegments];
//        for (int segment = 0; segment < numSegments; segment++) {
//            final double meanPosteriorCenter = meanPosteriorSummaries.get(segment).getCenter();
//            final double meanPosteriorStandardDeviation =
//                    (meanPosteriorSummaries.get(segment).getUpper() - meanPosteriorSummaries.get(segment).getLower()) / 2.;
//            meanPosteriorStandardDeviations[segment] = meanPosteriorStandardDeviation;
//            final double absoluteDifferenceFromTruth = Math.abs(meanPosteriorCenter - meansTruth.get(segment));
//            if (absoluteDifferenceFromTruth > meanPosteriorStandardDeviation) {
//                numMeansOutsideOneSigma++;
//            }
//            if (absoluteDifferenceFromTruth > 2 * meanPosteriorStandardDeviation) {
//                numMeansOutsideTwoSigma++;
//            }
//            if (absoluteDifferenceFromTruth > 3 * meanPosteriorStandardDeviation) {
//                numMeansOutsideThreeSigma++;
//            }
//        }
//        final double meanPosteriorStandardDeviationsMean = new Mean().evaluate(meanPosteriorStandardDeviations);
//        Assert.assertEquals(numMeansOutsideOneSigma, 100 - 68, DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_1_SIGMA);
//        Assert.assertEquals(numMeansOutsideTwoSigma, 100 - 95, DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_2_SIGMA);
//        Assert.assertTrue(numMeansOutsideThreeSigma <= DELTA_NUMBER_OF_MEANS_ALLOWED_OUTSIDE_3_SIGMA);
//        Assert.assertEquals(relativeError(meanPosteriorStandardDeviationsMean, MEAN_POSTERIOR_STANDARD_DEVIATION_MEAN_TRUTH),
//                0., RELATIVE_ERROR_THRESHOLD);
//
//        //check accuracy of latent outlier-indicator posterior samples
//        final List<CopyRatioState.OutlierIndicators> outlierIndicatorSamples =
//                modeller.getOutlierIndicatorsSamples();
//        int numIndicatorsCorrect = 0;
//        final int numIndicatorSamples = outlierIndicatorSamples.size();
//        final List<Integer> outlierIndicatorsTruthAsInt = new XReadLines(OUTLIER_INDICATORS_TRUTH_FILE)
//                .readLines().stream().map(Integer::parseInt).collect(Collectors.toList());
//        final List<Boolean> outlierIndicatorsTruth =
//                outlierIndicatorsTruthAsInt.stream().map(i -> i == 1).collect(Collectors.toList());
//        for (int index = 0; index < copyRatios.size(); index++) {
//            int numSamplesOutliers = 0;
//            for (final CopyRatioState.OutlierIndicators sample : outlierIndicatorSamples) {
//                if (sample.get(index)) {
//                    numSamplesOutliers++;
//                }
//            }
//            //take predicted state of indicator to be given by the majority of samples
//            if ((numSamplesOutliers >= numIndicatorSamples / 2.) == outlierIndicatorsTruth.get(index)) {
//                numIndicatorsCorrect++;
//            }
//        }
//        final double fractionOfOutlierIndicatorsCorrect = (double) numIndicatorsCorrect / copyRatios.size();
//        Assert.assertTrue(fractionOfOutlierIndicatorsCorrect >= FRACTION_OF_OUTLIER_INDICATORS_CORRECT_THRESHOLD);
//    }
}