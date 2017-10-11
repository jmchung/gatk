package org.broadinstitute.hellbender.utils.realignmentfilter;

import htsjdk.samtools.liftover.LiftOver;
import htsjdk.samtools.util.Interval;
import htsjdk.samtools.util.Locatable;
import org.broadinstitute.hellbender.utils.bwa.BwaMemAligner;
import org.broadinstitute.hellbender.utils.bwa.BwaMemAlignment;
import org.broadinstitute.hellbender.utils.bwa.BwaMemIndex;
import org.broadinstitute.hellbender.utils.read.GATKRead;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

public class Realigner {
    private final BwaMemAligner aligner;
    private final Optional<LiftOver> realignedToOriginalLiftover;
    private final Function<Interval, Interval> liftoverToOriginalCoordinates;


    public Realigner(final RealignmentFilterArgumentCollection rfac) {
        final BwaMemIndex index = new BwaMemIndex(rfac.bwaMemIndexImage);
        aligner = new BwaMemAligner(index);
        realignedToOriginalLiftover = rfac.liftoverChainFile == null ? Optional.empty() : Optional.of(new LiftOver(rfac.liftoverChainFile));
        liftoverToOriginalCoordinates = realignedToOriginalLiftover.isPresent() ? loc -> realignedToOriginalLiftover.get().liftOver(loc) : loc -> loc;

        aligner.setMinSeedLengthOption(rfac.minSeedLength);
        aligner.setDropRatioOption((float) rfac.dropRatio);
        aligner.setSplitFactorOption((float) rfac.splitFactor);
    }

    /**
     * Realign a collection of alt reads to see whether they are mapping artifacts
     * @param altReads Reads supposedly supporting a variant
     * @param ostensiblePosition Supposed location of these reads with respect to the bam's reference (not the realignment reference)
     */
    public void realign(final Collection<GATKRead> altReads, final Locatable ostensiblePosition) {
        final List<List<BwaMemAlignment>> alignments = aligner.alignSeqs(altReads, GATKRead::getBases);
    }

    private boolean mapsToSupposedLocation(final List<BwaMemAlignment> alignments, final Locatable supposedLocation) {
        if (alignments.isEmpty()) { // does this ever occur?
            return false;
        } else if (alignments.size() == 1) {
            final BwaMemAlignment alignment = alignments.get(0);
            alignment.getRefId()
            final Interval realignedInterval = new Interval(alignment.)
        }

    }
}
