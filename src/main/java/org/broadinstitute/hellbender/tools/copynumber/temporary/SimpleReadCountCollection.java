package org.broadinstitute.hellbender.tools.copynumber.temporary;

import com.opencsv.CSVReader;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.broadinstitute.hdf5.HDF5File;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.io.IOUtils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * //TODO replace this class with updated ReadCountCollection
 *
 * Simple class to enable input of a TSV or HDF5 file containing a list of {@link SimpleInterval} and integer read counts.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class SimpleReadCountCollection {
    private final String sampleName;
    private final List<SimpleInterval> intervals;
    private final RealMatrix readCounts;

    private SimpleReadCountCollection(final String sampleName,
                                      final List<SimpleInterval> intervals,
                                      final RealMatrix readCounts) {
        Utils.nonNull(sampleName);
        Utils.nonEmpty(intervals);
        Utils.nonNull(readCounts);
        Utils.validateArg(readCounts.getRowDimension() == 1, "Read-count matrix must contain only a single row.");
        Utils.validateArg(intervals.size() == readCounts.getColumnDimension(), "Number of intervals and read counts must match.");
        Utils.validateArg(Arrays.stream(readCounts.getRow(0)).noneMatch(c -> c < 0), "Read counts must all be non-negative integers.");
        Utils.validateArg(intervals.stream().distinct().count() == intervals.size(), "Intervals must all be unique.");

        this.sampleName = sampleName;
        this.intervals = intervals;
        this.readCounts = readCounts;
    }

    public String getSampleName() {
        return sampleName;
    }

    public List<SimpleInterval> getIntervals() {
        return intervals;
    }

    /**
     * @return single-row matrix containing the read counts
     */
    public RealMatrix getReadCounts() {
        return readCounts;
    }

    public static SimpleReadCountCollection read(final File file) {
        return TSVReader.read(file);
    }

    public static SimpleReadCountCollection read(final HDF5File file) {
        final HDF5ReadCountCollection hdf5ReadCountCollection = new HDF5ReadCountCollection(file);
        return new SimpleReadCountCollection(
                hdf5ReadCountCollection.getSampleName(),
                hdf5ReadCountCollection.getIntervals(),
                hdf5ReadCountCollection.getReadCounts());
    }

    private static final class TSVReader {
        private static final String COMMENT_STRING = "#";
        private static final char SEPARATOR_CHAR = '\t';

        private enum TSVColumn {
            CONTIG (0),
            START (1),
            STOP (2),
            NAME (3),
            READ_COUNT (4);

            private final int index;

            TSVColumn(final int index) {
                this.index = index;
            }
        }

        private static SimpleReadCountCollection read(final File file) {
            IOUtils.canReadFile(file);
            final int numRows = countRows(file);
            final List<SimpleInterval> intervals = new ArrayList<>(numRows);
            final List<Integer> readCounts = new ArrayList<>(numRows);
            try (final FileReader fileReader = new FileReader(file);
                 final CSVReader csvReader = new CSVReader(fileReader, SEPARATOR_CHAR)) {
                String[] row;
                while ((row = csvReader.readNext()) != null) {
                    if (!row[0].startsWith(COMMENT_STRING)) {  //skip comment lines
                        break;
                    }
                }
                final String sampleName = row != null ? row[TSVColumn.READ_COUNT.index] : null;
                while ((row = csvReader.readNext()) != null) {
                    final SimpleInterval interval = new SimpleInterval(
                            row[TSVColumn.CONTIG.index],
                            Integer.parseInt(row[TSVColumn.START.index]),
                            Integer.parseInt(row[TSVColumn.STOP.index]));
                    intervals.add(interval);
                    readCounts.add(Integer.parseInt(row[TSVColumn.READ_COUNT.index]));
                }
                final RealMatrix readCountsMatrix = new Array2DRowRealMatrix(1, readCounts.size());
                readCountsMatrix.setRow(0, readCounts.stream().mapToDouble(Integer::doubleValue).toArray());
                return new SimpleReadCountCollection(sampleName, intervals, readCountsMatrix);
            } catch (final IOException e) {
                throw new UserException.CouldNotReadInputFile(file);
            }
        }

        private static int countRows(final File file) {
            try {
                return (int) Files.lines(file.toPath()).filter(l -> !l.startsWith(COMMENT_STRING)).count() - 1;
            } catch (final IOException e) {
                throw new UserException.BadInput("Could not determine number of lines in TSV file.");
            }
        }
    }
}
