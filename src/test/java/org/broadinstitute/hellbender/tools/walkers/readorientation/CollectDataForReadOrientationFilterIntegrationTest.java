package org.broadinstitute.hellbender.tools.walkers.readorientation;

import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.SAMReadGroupRecord;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.utils.MathUtils;
import org.broadinstitute.hellbender.utils.genotyper.*;
import org.broadinstitute.hellbender.utils.read.ArtificialReadUtils;
import org.broadinstitute.hellbender.utils.read.GATKRead;
import org.broadinstitute.hellbender.utils.read.ReadUtils;
import org.broadinstitute.hellbender.utils.read.SAMFileGATKReadWriter;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.testng.internal.junit.ArrayAsserts;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by tsato on 8/1/17.
 */
public class CollectDataForReadOrientationFilterIntegrationTest extends CommandLineProgramTest {

    /**
     * Test the tool on a real bam to make sure that it does not crash
     */
    @Test
    public void testOnRealBam() {
        final File refTable = createTempFile("ref", ".table");
        final File altTable = createTempFile("alt", ".table");

        final String[] args = {
                "-R", v37_chr17_1Mb_Reference,
                "-I", NA12878_chr17_1k_BAM,
                "-alt_table", altTable.getAbsolutePath(),
                "-ref_table", refTable.getAbsolutePath()
        };

        runCommandLine(args);
    }

    @Test
    public void testOnSyntheticBam() throws IOException{
        final File refTable = createTempFile("ref", ".table");
        final File altTable = createTempFile("alt", ".table");

        final int numAltReads = 30;
        final int numRefReads = 70;
        final int depth = numAltReads + numRefReads;
        final File samFile = createSyntheticSam(numRefReads, numAltReads);

        final String[] args = {
                "-R", hg19_chr1_1M_Reference,
                "-I", samFile.getAbsolutePath(),
                "-alt_table", altTable.getAbsolutePath(),
                "-ref_table", refTable.getAbsolutePath()
        };

        runCommandLine(args);

        List<RefSiteHistogram> histograms = RefSiteHistogram.readRefSiteHistograms(refTable);
        List<AltSiteRecord> altDesignMatrix = AltSiteRecord.readAltSiteRecords(altTable);

        /** Expected result
         *
         * (position chr1: 100,000,000-100,000,011)
         *
         * ref: T C A C T A A G C A C A
         * alt: x C T C T G A C C A A x
         *          *     *   *     *
         *
         * At each alt site, we have 75 ref and 25 alt reads
         *
         * context, alt counts, alt f1r2 counts, depth
         * -------------------------------------------
         * CAC, {70, 0, 0, 30}, {35, 0, 0, 15}, 100
         * TAA, {70, 0, 30, 0}, {35, 0, 15, 0}, 100
         * AGC, {0, 30, 70, 0}, {0, 15, 35, 0}, 100
         * ACA, {30, 70, 0, 0}, {15, 35, 0, 0}, 100
         *
         *
         * Ref: 6 sites at depth 100
         * TCA, ACT, CTA, AAG, GCA, CAC
         **/

        // alt site tests
        Assert.assertEquals(altDesignMatrix.size(), 4);
        AltSiteRecord recordCAC = altDesignMatrix.stream().
                filter(rec -> rec.getReferenceContext().equals("CAC")).findFirst().get();
        ArrayAsserts.assertArrayEquals(recordCAC.getBaseCounts(), new int[]{70, 0, 0, 30});
        ArrayAsserts.assertArrayEquals(recordCAC.getF1R2Counts(), new int[]{35, 0, 0, 15});

        AltSiteRecord recordTAA = altDesignMatrix.stream().
                filter(rec -> rec.getReferenceContext().equals("TAA")).findFirst().get();
        ArrayAsserts.assertArrayEquals(recordTAA.getBaseCounts(), new int[]{70, 0, 30, 0});
        ArrayAsserts.assertArrayEquals(recordTAA.getF1R2Counts(), new int[]{35, 0, 15, 0});

        AltSiteRecord recordAGC = altDesignMatrix.stream().
                filter(rec -> rec.getReferenceContext().equals("AGC")).findFirst().get();
        ArrayAsserts.assertArrayEquals(recordAGC.getBaseCounts(), new int[]{0, 30, 70, 0});
        ArrayAsserts.assertArrayEquals(recordAGC.getF1R2Counts(), new int[]{0, 15, 35, 0});

        AltSiteRecord recordACA = altDesignMatrix.stream().
                filter(rec -> rec.getReferenceContext().equals("ACA")).findFirst().get();
        ArrayAsserts.assertArrayEquals(recordACA.getBaseCounts(), new int[]{30, 70, 0, 0});
        ArrayAsserts.assertArrayEquals(recordACA.getF1R2Counts(), new int[]{15, 35, 0, 0});

        // check ref histograms
        for (String exptectedRefContext : Arrays.asList("TCA", "ACT", "CTA", "AAG", "GCA", "CAC")){
            RefSiteHistogram histogram = histograms.stream()
                    .filter(hist -> hist.getReferenceContext().equals(exptectedRefContext))
                    .findFirst().get();
            // recall the refsite histogram places depth 1 at index 0
            // index: 0, 1, ..., depth-1, ...
            // depth: 1, 2, ..., depth,   ...
            Assert.assertEquals(histogram.getCounts()[depth-1], 1);
            Assert.assertEquals(MathUtils.sum(histogram.getCounts()), 1);
        }
    }

    private File createSyntheticSam(final int numRefReads, final int numAltReads) throws IOException {
        // create a sam header
        final int numChromosomes = 1;
        final int startingChromosome = 1;
        final int chromosomeSize = 1_000_000;
        final String readGroupName = "HELLO.1";
        final SAMFileHeader samHeader = ArtificialReadUtils.createArtificialSamHeader(
                numChromosomes, startingChromosome, chromosomeSize);
        samHeader.addReadGroup(new SAMReadGroupRecord(readGroupName));

        // create a sample list
        final int chromosomeIndex = 0;
        final String sampleName = "meganshand";
        final SampleList sampleList = new IndexedSampleList(sampleName);

        // specify chracteristics of reads
        final int depth = numAltReads + numRefReads;

        final int alignmentStart = 100_000;
        final byte[] refReadBases = "CACTAAGCAC".getBytes();
        final byte[] altReadBases = "CTCTGACCAA".getBytes(); // SNPs at indices 1, 4, 6, 9
        final int readLength = refReadBases.length;
        final byte baseq = 20;
        final byte[] quals = new byte[readLength];
        Arrays.fill(quals, baseq);
        final int mapq = 60;

        // create reads
        final List<GATKRead> reads = new ArrayList<>(depth);
        for (int i = 0; i < depth; i++) {
            final byte[] bases = i < numAltReads ? altReadBases : refReadBases;
            final GATKRead read = ArtificialReadUtils.createArtificialRead(samHeader, "Read" + i, chromosomeIndex,
                    alignmentStart, bases, quals, "10M");
            read.setReadGroup(readGroupName);
            read.setMappingQuality(mapq);
            read.setIsFirstOfPair();
            if (i % 2 == 0) {
                read.setIsReverseStrand(true);
            } else {
                read.setIsReverseStrand(false);
            }
            reads.add(read);
        }

        final File samFile = File.createTempFile("synthetic",".sam");
        final SAMFileGATKReadWriter writer = new SAMFileGATKReadWriter(
                ReadUtils.createCommonSAMWriter(samFile, null, samHeader, true, false, false));
        reads.forEach(r -> writer.addRead(r));
        writer.close(); // closing the writer writes to the file

        return samFile;
    }

}