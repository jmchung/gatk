package org.broadinstitute.hellbender.utils.realignmentfilter;

import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.SAMSequenceRecord;
import htsjdk.samtools.liftover.LiftOver;
import htsjdk.samtools.util.Interval;
import htsjdk.samtools.util.Locatable;
import org.apache.commons.lang3.StringUtils;
import org.broadinstitute.hellbender.utils.bwa.BwaMemAligner;
import org.broadinstitute.hellbender.utils.bwa.BwaMemAlignment;
import org.broadinstitute.hellbender.utils.bwa.BwaMemIndex;
import org.broadinstitute.hellbender.utils.read.GATKRead;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Realigner {
    private final BwaMemAligner aligner;
    private final Optional<LiftOver> realignedToOriginalLiftover;
    private final Function<Interval, Interval> liftoverToOriginalCoordinates;
    private final List<String> realignmentContigs;
    private final Map<String, String> bamToRealignmentContig;


    public Realigner(final RealignmentFilterArgumentCollection rfac, final SAMFileHeader bamHeader) {
        final BwaMemIndex index = new BwaMemIndex(rfac.bwaMemIndexImage);
        realignmentContigs = index.getReferenceContigNames();
        final List<String> bamContigs = bamHeader.getSequenceDictionary().getSequences().stream()
                .map(SAMSequenceRecord::getSequenceName).collect(Collectors.toList());

        bamToRealignmentContig = new HashMap<>();
        for (final String contig : bamContigs) {
            final List<String> possibleRelatedNames = Arrays.asList(contig, "chr" + contig, StringUtils.stripStart(contig, "chr"));
            possibleRelatedNames.stream().filter(realignmentContigs::contains).findFirst().ifPresent(ctg -> bamToRealignmentContig.put(contig, ctg));
        }

        aligner = new BwaMemAligner(index);
        realignedToOriginalLiftover = rfac.liftoverChainFile == null ? Optional.empty() : Optional.of(new LiftOver(rfac.liftoverChainFile));
        liftoverToOriginalCoordinates = realignedToOriginalLiftover.isPresent() ? loc -> realignedToOriginalLiftover.get().liftOver(loc) : loc -> loc;

        aligner.setMinSeedLengthOption(rfac.minSeedLength);
        aligner.setDropRatioOption((float) rfac.dropRatio);
        aligner.setSplitFactorOption((float) rfac.splitFactor);
    }


    public boolean mapsToSupposedLocation(final GATKRead read) {

        final List<BwaMemAlignment> alignments = aligner.alignSeqs(Arrays.asList(read), GATKRead::getBases).get(0);
        final Locatable supposedLocationWithRealignmentContig = new Interval(bamToRealignmentContig.get(read.getContig()), read.getStart(), read.getEnd());
        //TODO Incomplete!!!!!
        if (alignments.isEmpty()) { // does this ever occur?
            return false;
        } else if (alignments.size() == 1) {
            final BwaMemAlignment alignment = alignments.get(0);
            final int contigId = alignment.getRefId();
            if (contigId < 0) {
                return false;
            }
            final String realignedContig = realignmentContigs.get(alignment.getRefId());
            final Interval realignedInterval = new Interval(realignedContig, alignment.getRefStart(), alignment.getRefEnd());
            final Interval liftedBackInterval = liftoverToOriginalCoordinates.apply(realignedInterval);
            if (liftedBackInterval == null) {
                return false;
            } else if (liftedBackInterval.overlaps(read) || liftedBackInterval.overlaps(supposedLocationWithRealignmentContig)) {
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
