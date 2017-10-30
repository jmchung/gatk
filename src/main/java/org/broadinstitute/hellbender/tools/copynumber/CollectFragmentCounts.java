package org.broadinstitute.hellbender.tools.copynumber;

import com.google.common.annotations.VisibleForTesting;
import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.samtools.util.Locatable;
import htsjdk.samtools.util.OverlapDetector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.CopyNumberProgramGroup;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.ReadWalker;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.filters.MappingQualityReadFilter;
import org.broadinstitute.hellbender.engine.filters.ReadFilter;
import org.broadinstitute.hellbender.engine.filters.ReadFilterLibrary;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.copynumber.coverage.readcount.SimpleCount;
import org.broadinstitute.hellbender.tools.copynumber.coverage.readcount.SimpleCountCollection;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SampleMetadata;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SampleNameUtils;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SimpleSampleMetadata;
import org.broadinstitute.hellbender.utils.IntervalMergingRule;
import org.broadinstitute.hellbender.utils.IntervalSetRule;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.read.GATKRead;

import java.io.File;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Naive implementation of fragment-based coverage collection. The count for each interval is calculated by counting
 * how many different fragment centers intersect with this interval. The start and end positions of fragments are
 * inferred from read information. We only allow properly paired, first of pair reads - thus we do not double count
 * and we exclude reads whose fragment's position cannot be automatically inferred from its SAM record.
 *
 * @author Andrey Smirnov &lt;asmirnov@broadinstitute.org&gt;
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
@CommandLineProgramProperties(
        summary = "Collect fragment counts, by counting how many fragment centers intersect with a given interval. " +
                "The fragments are inferred from SAM records of only properly paired intervals.",
        oneLineSummary = "Collect fragment counts.",
        programGroup = CopyNumberProgramGroup.class
)
public final class CollectFragmentCounts extends ReadWalker {
    private static final Logger logger = LogManager.getLogger(CollectFragmentCounts.class);

    private static final int DEFAULT_MINIMUM_MAPPING_QUALITY = 30;

    private enum OutputFormat {
        TSV, HDF5
    }

    public static final String OUTPUT_FORMAT_LONG_NAME = "outputFormat";
    public static final String OUTPUT_FORMAT_SHORT_NAME = "fmt";

    @Argument(
            doc = "Output fragment-counts file",
            fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME,
            shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME
    )
    private File outputCountsFile = null;

    @Argument(
            doc = "Output file format.",
            fullName = OUTPUT_FORMAT_LONG_NAME,
            shortName = OUTPUT_FORMAT_SHORT_NAME,
            optional = true
    )
    private OutputFormat outputFormat = OutputFormat.HDF5;

    /**
     * Sample metadata contained in the BAM file.
     */
    private SampleMetadata sampleMetadata;

    private SAMSequenceDictionary sequenceDictionary;

    /**
     * Overlap detector used to determine when fragment centers overlap with input intervals.
     */
    private OverlapDetector<SimpleInterval> intervalOverlapDetector;

    private Map<SimpleInterval, Integer> intervalToCountsMap = new HashMap<>();

    @Override
    public List<ReadFilter> getDefaultReadFilters() {
        final List<ReadFilter> filters = new ArrayList<>(super.getDefaultReadFilters());
        filters.add(ReadFilterLibrary.MAPPED);
        filters.add(ReadFilterLibrary.NON_ZERO_REFERENCE_LENGTH_ALIGNMENT);
        filters.add(ReadFilterLibrary.NOT_DUPLICATE);
        filters.add(ReadFilterLibrary.FIRST_OF_PAIR); // this will make sure we don't double count
        filters.add(ReadFilterLibrary.PROPERLY_PAIRED);
        filters.add(new MappingQualityReadFilter(DEFAULT_MINIMUM_MAPPING_QUALITY));
        // this will only keep reads in pairs that are properly oriented and mapped on same chromosome
        // and lie within a few standard deviations from the mean of fragment size distributions
        return filters;
    }

    @Override
    public boolean requiresIntervals() {
        return true;
    }

    @Override
    public void onTraversalStart() {
        final String sampleName = SampleNameUtils.readSampleName(getHeaderForReads());
        sampleMetadata = new SimpleSampleMetadata(sampleName);

        //validate that the interval-argument collection parameters minimally modify the input intervals
        Utils.validateArg(intervalArgumentCollection.getIntervalSetRule() == IntervalSetRule.UNION,
                "Interval set rule must be set to UNION.");
        Utils.validateArg(intervalArgumentCollection.getIntervalExclusionPadding() == 0,
                "Interval exclusion padding must be set to 0.");
        Utils.validateArg(intervalArgumentCollection.getIntervalPadding() == 0,
                "Interval padding must be set to 0.");
        Utils.validateArg(intervalArgumentCollection.getIntervalMergingRule() == IntervalMergingRule.OVERLAPPING_ONLY,
                "Interval merging rule must be set to OVERLAPPING_ONLY.");

        sequenceDictionary = getBestAvailableSequenceDictionary();
        final List<SimpleInterval> intervals = intervalArgumentCollection.getIntervals(sequenceDictionary);
        intervalOverlapDetector = OverlapDetector.create(intervals);
        //verify again that intervals do not overlap
        Utils.validateArg(intervals.stream().noneMatch(i -> intervalOverlapDetector.getOverlaps(i).size() > 1),
                "Input intervals may not be overlapping.");

        //initialize interval-to-counts map
        intervals.forEach(i -> intervalToCountsMap.put(i, 0));

        logger.info("Collecting fragment counts...");
    }

    @Override
    public void apply(GATKRead read, ReferenceContext referenceContext, FeatureContext featureContext) {
        //getting a center of the fragment
        //TODO collect information on reads that do not have a properly paired mate
        final int centerOfFragment = ReadOrientation.getCenterOfFragment(read);

        //TODO make sure that center calculation always returns valid values
        if (1 <= centerOfFragment && centerOfFragment <= sequenceDictionary.getSequence(read.getContig()).getSequenceLength()) {
            final Locatable centerFragmentLocation = new SimpleInterval(read.getContig(), centerOfFragment, centerOfFragment);
            final Set<SimpleInterval> overlappingIntervals = intervalOverlapDetector.getOverlaps(centerFragmentLocation);
            if (overlappingIntervals.size() > 1) {
                // should not reach here since intervals are checked for overlapping;
                // doing a check to protect against future code changes
                throw new GATKException.ShouldNeverReachHereException("At most one interval can intersect with a center of a fragment.");
            }
            overlappingIntervals.forEach(i -> intervalToCountsMap.put(i, intervalToCountsMap.get(i) + 1));
        }
    }

    @Override
    public Object onTraversalSuccess() {
        logger.info("Writing fragment counts to " + outputCountsFile);
        final SimpleCountCollection fragmentCounts = new SimpleCountCollection(
                sampleMetadata,
                intervalToCountsMap.entrySet().stream()
                        .map(e -> new SimpleCount(e.getKey(), e.getValue()))
                        .collect(Collectors.toList()));

        if (outputFormat == OutputFormat.HDF5) {
            fragmentCounts.writeHDF5(outputCountsFile);
        } else {
            fragmentCounts.write(outputCountsFile);
        }

        return "SUCCESS";
    }

    /**
     * Helper class to calculate fragment center of a properly paired read
     */
    @VisibleForTesting
    protected enum ReadOrientation {

        /**
         * Read was located on forward strand
         */
        FORWARD(read -> read.getUnclippedStart() + read.getFragmentLength() / 2),

        /**
         * Read was located on reverse strand
         */
        REVERSE(read -> read.getUnclippedStart() + (read.getLength() - 1)  + read.getFragmentLength() / 2);

        private final Function<GATKRead, Integer> readToFragmentCenterMapper;

        ReadOrientation(final Function<GATKRead, Integer> readToCenterMapper) {
            this.readToFragmentCenterMapper = readToCenterMapper;
        }

        /**
         * Get a function that maps the read to the center of the fragment
         */
        protected Function<GATKRead, Integer> getReadToFragmentCenterMapper() {
            return readToFragmentCenterMapper;
        }

        /**
         * Get {@link ReadOrientation} instance corresponding to the orientation of the read
         */
        protected static ReadOrientation getReadOrientation(final GATKRead read) {
            return read.getFragmentLength() > 0 ? FORWARD : REVERSE;
        }

        /**
         * Compute center of the fragment that read corresponds to
         */
        protected static int getCenterOfFragment(final GATKRead read) {
            return getReadOrientation(read).getReadToFragmentCenterMapper().apply(read);
        }
    }
}
