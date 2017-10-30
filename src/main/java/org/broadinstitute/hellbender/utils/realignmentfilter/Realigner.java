package org.broadinstitute.hellbender.utils.realignmentfilter;

import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.liftover.LiftOver;
import htsjdk.samtools.util.Interval;
import org.broadinstitute.hellbender.utils.bwa.BwaMemAligner;
import org.broadinstitute.hellbender.utils.bwa.BwaMemAlignment;
import org.broadinstitute.hellbender.utils.bwa.BwaMemIndex;
import org.broadinstitute.hellbender.utils.read.GATKRead;

import java.util.*;
import java.util.function.Function;

public class Realigner {
    private final BwaMemAligner aligner;
    private final Optional<LiftOver> liftoverToRealignmentCoordinates;
    private final Function<Interval, Interval> convertToRealignmentCoordinates;
    private final List<String> realignmentContigs;

    public Realigner(final RealignmentFilterArgumentCollection rfac, final SAMFileHeader bamHeader) {
        final BwaMemIndex index = new BwaMemIndex(rfac.bwaMemIndexImage);
        realignmentContigs = index.getReferenceContigNames();
        aligner = new BwaMemAligner(index);
        liftoverToRealignmentCoordinates = rfac.liftoverChainFile == null ? Optional.empty() : Optional.of(new LiftOver(rfac.liftoverChainFile));
        convertToRealignmentCoordinates = liftoverToRealignmentCoordinates.isPresent() ? loc -> liftoverToRealignmentCoordinates.get().liftOver(loc) : loc -> loc;

        aligner.setMinSeedLengthOption(rfac.minSeedLength);
        aligner.setDropRatioOption((float) rfac.dropRatio);
        aligner.setSplitFactorOption((float) rfac.splitFactor);
    }

    public Interval getRealignemntCoordinates(final Interval bamCoordinates) {
        return convertToRealignmentCoordinates.apply(bamCoordinates);
    }


    public boolean mapsToSupposedLocation(final GATKRead read, final Interval supposedRealignmentLocation) {

        final List<BwaMemAlignment> alignments = aligner.alignSeqs(Arrays.asList(read), GATKRead::getBases).get(0);
        if (supposedRealignmentLocation == null) {
            return false;
        }
        //TODO Incomplete!!!!!
        if (alignments.isEmpty()) { // does this ever occur?
            return false;
        } else if (alignments.size() == 1) {
            final BwaMemAlignment alignment = alignments.get(0);
            final int contigId = alignment.getRefId();
            if (contigId < 0) {
                return false;
            }
            if (new Interval(realignmentContigs.get(contigId), alignment.getRefStart(), alignment.getRefEnd()).overlaps(supposedRealignmentLocation)) {
                return true;
            } else {
                int j = 4;
                return false;
            }
        } else {
            int q = 3;
            //TODO: flesh out
            return false;
        }

    }
}
