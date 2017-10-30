package org.broadinstitute.hellbender.utils.realignmentfilter;

import org.broadinstitute.barclay.argparser.Argument;

import java.io.File;

public class RealignmentFilterArgumentCollection {

    public static final int DEFAULT_MIN_SEED_LENGTH = 16;
    public static final double DEFAULT_DROP_RATIO = 0.1;
    public static final double DEFAULT_SEED_SPLIT_FACTOR = 1.2;

    /**
     * BWA-mem index image created by {@link org.broadinstitute.hellbender.tools.BwaMemIndexImageCreator}
     */
    @Argument(fullName = "bwa-mem-index-image", shortName = "index", doc = "BWA-mem index image", optional = true)
    public String bwaMemIndexImage;

    /**
     * UCSC chain file for liftover of reads realigned to realignment reference, in case this differs from the original reference to which the bam was aligned.
     */
    @Argument(fullName = "liftover-chain-file", shortName = "chain", optional = true, doc = "UCSC chain file for liftover of realigned coordinates to original coordinates in case")
    public File liftoverChainFile;

    /**
     * BWA-mem minimum seed length
     */
    @Argument(fullName = "minimum-seed-length", shortName = "min-seed-length", optional = true, doc = "Minimum number of matching bases to seed a MEM")
    public int minSeedLength = DEFAULT_MIN_SEED_LENGTH;

    /**
     * BWA-mem extension drop ratio
     */
    @Argument(fullName = "drop-ratio", shortName = "drop-ratio", doc = "Fraction of best MEM extension score below which other extensions are dropped")
    public double dropRatio = DEFAULT_DROP_RATIO;

    /**
     * BWA-mem seed split factor
     */
    @Argument(fullName = "seed-split-factor", shortName = "split-factor", doc = "MEMs longer than the minimum seed length times this factor are split and re-seeded.")
    public double splitFactor = DEFAULT_SEED_SPLIT_FACTOR;

}
