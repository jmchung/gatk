package org.broadinstitute.hellbender.tools.copynumber.coverage.segmentation;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.tools.copynumber.coverage.copyratio.CopyRatio;
import org.broadinstitute.hellbender.tools.copynumber.coverage.copyratio.CopyRatioCollection;
import org.broadinstitute.hellbender.tools.copynumber.utils.segmentation.KernelSegmenter;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.param.ParamUtils;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * TODO
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class CopyRatioKernelSegmenter {
    private static final Logger logger = LogManager.getLogger(CopyRatioKernelSegmenter.class);

    private static final int MIN_NUM_POINTS_REQUIRED_PER_CHROMOSOME = 10;

    //Gaussian kernel for a specified variance; if variance is zero, use a linear kernel
    private static final Function<Double, BiFunction<Double, Double, Double>> KERNEL =
            variance -> variance == 0.
                    ? (x, y) -> x * y
                    : (x, y) -> FastMath.exp(-(x - y) * (x - y) / (2. * variance));

    private final CopyRatioCollection denoisedCopyRatios;
    private final Map<String, List<SimpleInterval>> intervalsPerChromosome;
    private final Map<String, List<Double>> denoisedCopyRatiosPerChromosome;    //in log2 space

    /**
     * @param denoisedCopyRatios  in log2 space
     */
    public CopyRatioKernelSegmenter(final CopyRatioCollection denoisedCopyRatios) {
        Utils.nonNull(denoisedCopyRatios);
        this.denoisedCopyRatios = denoisedCopyRatios;
        intervalsPerChromosome = denoisedCopyRatios.getRecords().stream().map(CopyRatio::getInterval)
                .collect(Collectors.groupingBy(
                        SimpleInterval::getContig,
                        LinkedHashMap::new,
                        Collectors.toList()));
        final List<Double> denoisedCopyRatioValues = denoisedCopyRatios.getLog2CopyRatioValues();
        denoisedCopyRatiosPerChromosome = IntStream.range(0, denoisedCopyRatios.getRecords().size()).boxed()
                .map(i -> new ImmutablePair<>(
                        denoisedCopyRatios.getRecords().get(i).getContig(),
                        denoisedCopyRatioValues.get(i)))
                .collect(Collectors.groupingBy(
                        Pair::getKey,
                        LinkedHashMap::new,
                        Collectors.mapping(Pair::getValue, Collectors.toList())));
    }

    public CopyRatioSegmentCollection findSegmentation(final int maxNumChangepointsPerChromosome,
                                                       final double kernelVariance,
                                                       final int kernelApproximationDimension,
                                                       final List<Integer> windowSizes,
                                                       final double numChangepointsPenaltyLinearFactor,
                                                       final double numChangepointsPenaltyLogLinearFactor) {
        ParamUtils.isPositiveOrZero(maxNumChangepointsPerChromosome, "Maximum number of changepoints must be non-negative.");
        ParamUtils.isPositiveOrZero(kernelVariance, "Variance of Gaussian kernel must be non-negative (if zero, a linear kernel will be used).");
        ParamUtils.isPositive(kernelApproximationDimension, "Dimension of kernel approximation must be positive.");
        Utils.validateArg(windowSizes.stream().allMatch(ws -> ws > 0), "Window sizes must all be positive.");
        Utils.validateArg(new HashSet<>(windowSizes).size() == windowSizes.size(), "Window sizes must all be unique.");
        ParamUtils.isPositiveOrZero(numChangepointsPenaltyLinearFactor,
                "Linear factor for the penalty on the number of changepoints per chromosome must be non-negative.");
        ParamUtils.isPositiveOrZero(numChangepointsPenaltyLogLinearFactor,
                "Log-linear factor for the penalty on the number of changepoints per chromosome must be non-negative.");

        logger.info(String.format("Finding changepoints in %d data points and %d chromosomes...",
                denoisedCopyRatios.getRecords().size(), denoisedCopyRatiosPerChromosome.size()));

        //loop over chromosomes, find changepoints, and create copy-ratio segments
        final List<CopyRatioSegment> segments = new ArrayList<>();
        for (final String chromosome : denoisedCopyRatiosPerChromosome.keySet()) {
            final List<Double> denoisedCopyRatiosInChromosome = denoisedCopyRatiosPerChromosome.get(chromosome);
            final int numDenoisedCopyRatiosInChromosome = denoisedCopyRatiosInChromosome.size();
            logger.info(String.format("Finding changepoints in %d data points in chromosome %s...",
                    numDenoisedCopyRatiosInChromosome, chromosome));

            if (numDenoisedCopyRatiosInChromosome < MIN_NUM_POINTS_REQUIRED_PER_CHROMOSOME) {
                logger.warn(String.format("Number of points in chromosome %s (%d) is less than that required (%d), skipping segmentation...",
                        chromosome, numDenoisedCopyRatiosInChromosome, MIN_NUM_POINTS_REQUIRED_PER_CHROMOSOME));
                final int start = intervalsPerChromosome.get(chromosome).get(0).getStart();
                final int end = intervalsPerChromosome.get(chromosome).get(numDenoisedCopyRatiosInChromosome - 1).getEnd();
                segments.add(new CopyRatioSegment(
                        new SimpleInterval(chromosome, start, end), denoisedCopyRatiosInChromosome));
            }

            final List<Integer> changepoints = new ArrayList<>(new KernelSegmenter<>(denoisedCopyRatiosInChromosome)
                .findChangepoints(maxNumChangepointsPerChromosome, KERNEL.apply(kernelVariance), kernelApproximationDimension,
                        windowSizes, numChangepointsPenaltyLinearFactor, numChangepointsPenaltyLogLinearFactor, KernelSegmenter.ChangepointSortOrder.INDEX));

            if (!changepoints.contains(numDenoisedCopyRatiosInChromosome)) {
                changepoints.add(numDenoisedCopyRatiosInChromosome - 1);
            }
            int previousChangepoint = -1;
            for (final int changepoint : changepoints) {
                final int start = intervalsPerChromosome.get(chromosome).get(previousChangepoint + 1).getStart();
                final int end = intervalsPerChromosome.get(chromosome).get(changepoint).getEnd();
                final List<Double> denoisedCopyRatiosInSegment = denoisedCopyRatiosInChromosome.subList(
                        previousChangepoint + 1, changepoint + 1);
                segments.add(new CopyRatioSegment(
                        new SimpleInterval(chromosome, start, end),
                        denoisedCopyRatiosInSegment));
                previousChangepoint = changepoint;
            }
        }
        logger.info(String.format("Found %d segments in %d chromosomes.", segments.size(), denoisedCopyRatiosPerChromosome.keySet().size()));
        return new CopyRatioSegmentCollection(denoisedCopyRatios.getSampleMetadata(), segments);
    }
}
