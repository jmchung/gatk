package org.broadinstitute.hellbender.tools.copynumber.allelic.segmentation;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.tools.copynumber.allelic.alleliccount.AllelicCount;
import org.broadinstitute.hellbender.tools.copynumber.allelic.alleliccount.AllelicCountCollection;
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
public final class AlleleFractionKernelSegmenter {
    private static final Logger logger = LogManager.getLogger(AlleleFractionKernelSegmenter.class);

    private static final int MIN_NUM_POINTS_REQUIRED_PER_CHROMOSOME = 10;

    //Gaussian kernel for a specified variance; if variance is zero, use a linear kernel
    private static final Function<Double, BiFunction<Double, Double, Double>> KERNEL =
            variance -> variance == 0.
                    ? (x, y) -> x * y
                    : (x, y) -> FastMath.exp(-(x - y) * (x - y) / (2. * variance));

    private final AllelicCountCollection allelicCounts;
    private final Map<String, List<SimpleInterval>> intervalsPerChromosome;
    private final Map<String, List<AllelicCount>> allelicCountsPerChromosome;
    private final Map<String, List<Double>> alternateAlleleFractionsPerChromosome;

    public AlleleFractionKernelSegmenter(final AllelicCountCollection allelicCounts) {
        Utils.nonNull(allelicCounts);
        this.allelicCounts = allelicCounts;
        intervalsPerChromosome = allelicCounts.getRecords().stream()
                .map(AllelicCount::getInterval)
                .collect(Collectors.groupingBy(
                        SimpleInterval::getContig,
                        LinkedHashMap::new,
                        Collectors.toList()));
        allelicCountsPerChromosome = IntStream.range(0, allelicCounts.getRecords().size()).boxed()
                .map(i -> new ImmutablePair<>(
                        allelicCounts.getRecords().get(i).getContig(),
                        allelicCounts.getRecords().get(i)))
                .collect(Collectors.groupingBy(
                        Pair::getKey,
                        LinkedHashMap::new,
                        Collectors.mapping(Pair::getValue, Collectors.toList())));
        final double[] alternateAlleleFractions = allelicCounts.getRecords().stream()
                .mapToDouble(AllelicCount::getAlternateAlleleFraction)
                .toArray();
        alternateAlleleFractionsPerChromosome = IntStream.range(0, allelicCounts.getRecords().size()).boxed()
                .map(i -> new ImmutablePair<>(
                        allelicCounts.getRecords().get(i).getContig(),
                        alternateAlleleFractions[i]))
                .collect(Collectors.groupingBy(
                        Pair::getKey,
                        LinkedHashMap::new,
                        Collectors.mapping(Pair::getValue, Collectors.toList())));
    }

    public AlleleFractionSegmentCollection findSegmentation(final int maxNumChangepointsPerChromosome,
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
                allelicCounts.getRecords().size(), alternateAlleleFractionsPerChromosome.size()));

        //loop over chromosomes, find changepoints, and create allele-fraction segments
        final List<AlleleFractionSegment> segments = new ArrayList<>();
        for (final String chromosome : alternateAlleleFractionsPerChromosome.keySet()) {
            final List<AllelicCount> allelicCountsInChromosome = allelicCountsPerChromosome.get(chromosome);
            final List<Double> alternateAlleleFractionsInChromosome = alternateAlleleFractionsPerChromosome.get(chromosome);
            final int numAllelicCountsInChromosome = allelicCountsInChromosome.size();
            logger.info(String.format("Finding changepoints in %d data points in chromosome %s...",
                    numAllelicCountsInChromosome, chromosome));

            if (numAllelicCountsInChromosome < MIN_NUM_POINTS_REQUIRED_PER_CHROMOSOME) {
                logger.warn(String.format("Number of points in chromosome %s (%d) is less than that required (%d), skipping segmentation...",
                        chromosome, numAllelicCountsInChromosome, MIN_NUM_POINTS_REQUIRED_PER_CHROMOSOME));
                final int start = allelicCountsInChromosome.get(0).getStart();
                final int end = allelicCountsInChromosome.get(numAllelicCountsInChromosome - 1).getEnd();
                segments.add(new AlleleFractionSegment(
                        new SimpleInterval(chromosome, start, end), numAllelicCountsInChromosome));
            }

            final List<Integer> changepoints = new ArrayList<>(new KernelSegmenter<>(alternateAlleleFractionsInChromosome)
                .findChangepoints(maxNumChangepointsPerChromosome, KERNEL.apply(kernelVariance), kernelApproximationDimension,
                        windowSizes, numChangepointsPenaltyLinearFactor, numChangepointsPenaltyLogLinearFactor, KernelSegmenter.ChangepointSortOrder.INDEX));

            if (!changepoints.contains(numAllelicCountsInChromosome)) {
                changepoints.add(numAllelicCountsInChromosome - 1);
            }
            int previousChangepoint = -1;
            for (final int changepoint : changepoints) {
                final int start = intervalsPerChromosome.get(chromosome).get(previousChangepoint + 1).getStart();
                final int end = intervalsPerChromosome.get(chromosome).get(changepoint).getEnd();
                final List<AllelicCount> allelicCountsInSegment = allelicCountsInChromosome.subList(
                        previousChangepoint + 1, changepoint + 1);
                segments.add(new AlleleFractionSegment(
                        new SimpleInterval(chromosome, start, end), allelicCountsInSegment));
                previousChangepoint = changepoint;
            }
        }
        logger.info(String.format("Found %d segments in %d chromosomes.", segments.size(), alternateAlleleFractionsPerChromosome.keySet().size()));
        return new AlleleFractionSegmentCollection(allelicCounts.getSampleMetadata(), segments);
    }
}
