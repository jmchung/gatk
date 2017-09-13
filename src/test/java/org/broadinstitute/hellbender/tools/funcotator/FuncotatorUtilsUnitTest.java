package org.broadinstitute.hellbender.tools.funcotator;

import htsjdk.samtools.util.Locatable;
import htsjdk.variant.variantcontext.Allele;
import lombok.Data;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.ReferenceFileSource;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A unit test suite for the {@link FuncotatorUtils} class.
 * Created by jonn on 9/1/17.
 */
public class FuncotatorUtilsUnitTest extends BaseTest {

    //==================================================================================================================
    // Static Variables:
    private static final File TEST_REFERENCE = new File(hg19MiniReference);
    private static final String TEST_REFERENCE_CONTIG = "1";
    private static final int TEST_REFERENCE_START = 12000;
    private static final int TEST_REFERENCE_END = 16000;

    //==================================================================================================================
    // Helper Methods:

    /**
     * Prints the bases of the {@link FuncotatorUtilsUnitTest#TEST_REFERENCE} from {@link FuncotatorUtilsUnitTest#TEST_REFERENCE_START} to {@link FuncotatorUtilsUnitTest#TEST_REFERENCE_END}
     * The print out has each base numbered - the results must be interpreted vertically (i.e. read the numbers from top to bottom to get the index of the base).
     * For example, the first 21 bases are as follows (with labels for the example):
     *
     *   Base 5       Base 19
     *     |             |
     * 000000000000000000000
     * 000000000000000000000
     * 000000000011111111112
     * 123456789012345678901
     * TCATCTGCAGGTGTCTGACTT
     *
     */
    private void printReferenceBases() {
        printReferenceBases(TEST_REFERENCE, TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END);
    }

    /**
     * Writes the bases of the {@link FuncotatorUtilsUnitTest#TEST_REFERENCE} from {@link FuncotatorUtilsUnitTest#TEST_REFERENCE_START} to {@link FuncotatorUtilsUnitTest#TEST_REFERENCE_END} to a file.
     * The print out has each base numbered - the results must be interpreted vertically (i.e. read the numbers from top to bottom to get the index of the base).
     * For example, the first 21 bases are as follows (with labels for the example):
     *
     *   Base 5       Base 19
     *     |             |
     * 000000000000000000000
     * 000000000000000000000
     * 000000000011111111112
     * 012345678901234567890
     * TCATCTGCAGGTGTCTGACTT
     *
     * @param contig Contig from which to print bases.
     * @param start Start point in contig from which to print bases.
     * @param end End point in contig to which to print bases.
     */
    private void printReferenceBases(final File refFile, final String contig, final int start, final int end) {
        final ReferenceContext ref = new ReferenceContext(new ReferenceFileSource(refFile), new SimpleInterval(contig, start, end));

        // Ones place:
        final StringBuilder sb_o = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length; ++i ) {
            sb_o.append(i % 10);
        }
        // Tens place:
        final StringBuilder sb_t = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length; ++i ) {
            sb_t.append((int)(i / 10.0) % 10);
        }
        // Hundreds place:
        final StringBuilder sb_h = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length; ++i ) {
            sb_h.append((int)(i / 100.0) % 10);
        }
        // Thousands place:
        final StringBuilder sb_th = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length; ++i ) {
            sb_th.append((int)(i / 1000.0) % 10);
        }
        // Ten Thousands place:
        final StringBuilder sb_tth = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length; ++i ) {
            sb_tth.append((int)(i / 10000.0) % 10);
        }

        try (Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("refBases_"+contig+"_"+start+"-"+end+".txt")))) {
            writer.write("Location: " + contig + ":" + start + ":" + end + "\n");
            writer.write("=================================================================================\n");
            writer.write( sb_tth.toString() + "\n");
            writer.write( sb_th.toString() + "\n");
            writer.write( sb_h.toString() + "\n" );
            writer.write( sb_t.toString() + "\n" );
            writer.write( sb_o.toString() + "\n" );
            writer.write( new String(ref.getBases()) + "\n\n" );
        }
        catch ( final IOException ex ) {
            throw new GATKException("Could not create an output file!", ex);
        }

        System.out.println();
        System.out.println("Location: " + contig + ":" + start + ":" + end);
        System.out.println("=================================================================================");
        System.out.println( sb_tth.toString() );
        System.out.println( sb_th.toString() );
        System.out.println( sb_h.toString() );
        System.out.println( sb_t.toString() );
        System.out.println( sb_o.toString() );
        System.out.println( new String(ref.getBases()) );
    }

//    @Test
//    void createRefBaseFile() {
//        printReferenceBases(new File("/Users/jonn/Development/references/GRCh37.p13.genome.fasta"), "chr1", 860000,  880000);
//        printReferenceBases();
//    }

    //==================================================================================================================
    // Data Providers:

    @DataProvider
    Object[][] provideReferenceAndExonListAndExpected() {

        return new Object[][] {
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.emptyList(),
                        ""
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)),
                        "GCAGAGACGGGAGGGGCAGAGCCGCAGGCACAGCCAAGAGGGCTGAAGAAA"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Arrays.asList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550),
                                new SimpleInterval("1", TEST_REFERENCE_START + 551, TEST_REFERENCE_START + 600)
                        ),
                        "GCAGAGACGGGAGGGGCAGAGCCGCAGGCACAGCCAAGAGGGCTGAAGAAATGGTAGAACGGAGCAGCTGGTGATGTGTGGGCCCACCGGCCCCAGGCTCC"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Arrays.asList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500),
                                new SimpleInterval("1", TEST_REFERENCE_START + 501, TEST_REFERENCE_START + 501),
                                new SimpleInterval("1", TEST_REFERENCE_START + 502, TEST_REFERENCE_START + 502),
                                new SimpleInterval("1", TEST_REFERENCE_START + 503, TEST_REFERENCE_START + 503),
                                new SimpleInterval("1", TEST_REFERENCE_START + 504, TEST_REFERENCE_START + 504),
                                new SimpleInterval("1", TEST_REFERENCE_START + 505, TEST_REFERENCE_START + 505),
                                new SimpleInterval("1", TEST_REFERENCE_START + 506, TEST_REFERENCE_START + 506),
                                new SimpleInterval("1", TEST_REFERENCE_START + 507, TEST_REFERENCE_START + 507),
                                new SimpleInterval("1", TEST_REFERENCE_START + 508, TEST_REFERENCE_START + 508),
                                new SimpleInterval("1", TEST_REFERENCE_START + 509, TEST_REFERENCE_START + 509)
                        ),
                        "GCAGAGACGG"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500)
                        ),
                        "G"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, 1, 10)),
                        Collections.singletonList(
                                new SimpleInterval("1", 1, 1)
                        ),
                        "N"
                },
        };
    }

    @DataProvider
    Object[][] provideReferenceAndExonListForGatkExceptions() {

        return new Object[][] {
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("2", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)
                        ),
                },
        };
    }

    @DataProvider
    Object[][] provideReferenceAndExonListForIllegalArgumentExceptions() {

        return new Object[][] {
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START)
                        ),
                },
        };
    }

    @DataProvider
    Object[][] provideDataForGetStartPositionInTranscript() {

        final List<? extends Locatable> exons = Arrays.asList(
                new SimpleInterval("chr1", 10,19),
                new SimpleInterval("chr1", 30,39),
                new SimpleInterval("chr1", 50,59),
                new SimpleInterval("chr1", 70,79),
                new SimpleInterval("chr1", 90,99)
        );

        return new Object[][] {
                { new SimpleInterval("chr1", 1, 1), exons, -1 },
                { new SimpleInterval("chr1", 25, 67), exons, -1 },
                { new SimpleInterval("chr1", 105, 392), exons, -1 },
                { new SimpleInterval("chr1", 10, 10), exons, 0 },
                { new SimpleInterval("chr1", 99, 99), exons, 49 },
                { new SimpleInterval("chr1", 50, 67), exons, 20 },
        };
    }

    @DataProvider
    Object[][] provideAllelesAndFrameshiftResults() {
        return new Object[][] {
                { Allele.create((byte)'A'), Allele.create((byte)'A'), false },
                { Allele.create((byte)'A'), Allele.create((byte)'T'), false },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        false
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'T'}),
                        false
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'T',(byte)'T'}),
                        false
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A'}),
                        false
                },
                {
                        Allele.create(new byte[] {(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A',(byte)'A'}),
                        false
                },

                // ======================
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        true
                },

                {
                        Allele.create(new byte[] {(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A',(byte)'A'}),
                        true
                },
                {
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A'}),
                        Allele.create(new byte[] {(byte)'A',(byte)'A',(byte)'A',(byte)'A'}),
                        true
                },
        };
    }

    @DataProvider
    Object[][] providePositionAndExpectedAlignedPosition() {
        return new Object[][] {
                {0,0},
                {1,0},
                {2,0},
                {3,3},
                {4,3},
                {5,3},
                {1635,1635},
                {1636,1635},
                {1637,1635},
        };
    }

    @DataProvider
    Object[][] providePositionAndExpectedAlignedEndPosition() {
        return new Object[][] {
                {0,1,2},
                {0,2,2},
                {0,3,2},
                {0,4,5},
                {0,5,5},
                {0,6,5},
        };
    }

    @DataProvider
    Object[][] provideDataForGetAlternateCodingSequence() {
        return new Object[][] {
                {
                    "01234567890A1234567890123456789", 12, Allele.create((byte)'A'), Allele.create((byte)'A'), "01234567890A1234567890123456789"
                },
                {
                    "01234567890A1234567890123456789", 11, Allele.create((byte)'A'), Allele.create((byte)'A'), "0123456789AA1234567890123456789"
                },
                {
                    "01234567890A1234567890123456789", 12, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "01234567890ATGCATGC1234567890123456789"
                },
                {
                    "AAAAATTTTTGGGGGCCCCCAAAAATTTTTGGGGGCCCCC", 12, Allele.create("GGGGCCC".getBytes()), Allele.create((byte)'T'), "AAAAATTTTTGTCCAAAAATTTTTGGGGGCCCCC"
                },
                {
                    "A", 1, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "ATGCATGC"
                },
                {
                    "BA", 2, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "BATGCATGC"
                },
                {
                    "AB", 1, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "ATGCATGCB"
                },
        };
    }

    @DataProvider
    Object[][] provideDataForGetEukaryoticAminoAcidByCodon() {
        return new Object[][] {
                {null, null},
                {"", null},
                {"XQZ", null},
                {"ATG", AminoAcid.METHIONINE},
                {"CCA", AminoAcid.PROLINE},
                {"CCC", AminoAcid.PROLINE},
                {"CCG", AminoAcid.PROLINE},
                {"CCT", AminoAcid.PROLINE},
        };
    }

    @DataProvider
    Object[][] provideDataForGetMitochondrialAminoAcidByCodon() {
        return new Object[][]{
                {null, false, null},
                {"", false, null},
                {"XQZ", false, null},
                {null, true, null},
                {"", true, null},
                {"XQZ", true, null},
                {"ATG", false, AminoAcid.METHIONINE},
                {"CCA", false, AminoAcid.PROLINE},
                {"CCC", false, AminoAcid.PROLINE},
                {"CCG", false, AminoAcid.PROLINE},
                {"CCT", false, AminoAcid.PROLINE},
                {"ATT", false, AminoAcid.ISOLEUCINE},
                {"ATT", true, AminoAcid.METHIONINE},
                {"ATA", false, AminoAcid.METHIONINE},
                {"AGA", false, AminoAcid.STOP_CODON},
                {"AGG", false, AminoAcid.STOP_CODON},
                {"TGA", false, AminoAcid.TRYPTOPHAN},
        };
    }

    @DataProvider
    Object[][] provideDataForIsInFrameWithEndOfRegion() {
        return new Object[][] {

                // Starting position checks:
                { 1, 1, false },
                { 1, 2, false },
                { 1, 3, true  },
                { 1, 4, false },
                { 1, 5, false },
                { 1, 6, true  },
                { 1, 7, false },
                { 1, 8, false },
                { 1, 9, true  },
                { 1, 10, false },

                // Middle position checks:
                { 1, 10, false },
                { 2, 10, true },
                { 3, 10, false },
                { 4, 10, false },
                { 5, 10, true },
                { 6, 10, false },
                { 7, 10, false },
                { 8, 10, true },
                { 9, 10, false },
                { 10, 10, false },
                { 56, 473, false },
                { 57, 473, true },
                { 58, 473, false },

                // Last position should always be out of frame.
                { 10, 10, false },
                { 11, 11, false },
                { 12, 12, false },
                { 100, 100, false },
                { 1000, 1000, false },
        };
    }

    //==================================================================================================================
    // Tests:

    @Test(dataProvider = "provideAllelesAndFrameshiftResults")
    void testIsFrameshift(final Allele ref, final Allele alt, final boolean expected) {
        Assert.assertEquals( FuncotatorUtils.isFrameshift(ref, alt), expected );
    }

    @Test(dataProvider = "provideReferenceAndExonListAndExpected")
    void testGetCodingSequence(final ReferenceContext reference, final List<Locatable> exonList, final String expected) {
        Assert.assertEquals( FuncotatorUtils.getCodingSequence(reference, exonList), expected );
    }

    @Test(dataProvider = "provideReferenceAndExonListForGatkExceptions",
            expectedExceptions = GATKException.class)
    void testGetCodingSequenceWithGatkExceptions(final ReferenceContext reference, final List<Locatable> exonList) {
        FuncotatorUtils.getCodingSequence(reference, exonList);
    }

//    @Test(dataProvider = "provideReferenceAndExonListForIllegalArgumentExceptions",
//            expectedExceptions = IllegalArgumentException.class)
//    void testGetCodingSequenceWithIllegalArgumentExceptions(final ReferenceContext reference, final List<Locatable> exonList) {
//        FuncotatorUtils.getCodingSequence(reference, exonList);
//    }

    @Test(dataProvider = "provideDataForGetStartPositionInTranscript")
    void testGetStartPositionInTranscript(final Locatable variant, final List<? extends Locatable> transcript, final int expected) {
        Assert.assertEquals( FuncotatorUtils.getStartPositionInTranscript(variant, transcript), expected );
    }

    @Test(dataProvider = "providePositionAndExpectedAlignedPosition")
    void testGetAlignedPosition(final int pos, final int expected) {
        Assert.assertEquals(FuncotatorUtils.getAlignedPosition(pos), expected);
    }

    @Test(dataProvider = "providePositionAndExpectedAlignedEndPosition")
    void testGetAlignedEndPosition(final int alignedStart, final int length, final int expected) {
        Assert.assertEquals(FuncotatorUtils.getAlignedEndPosition(alignedStart, length), expected);
    }

    @Test(dataProvider = "provideDataForGetAlternateCodingSequence")
    void testGetAlternateCodingSequence(final String refCodingSeq, final int startPos, final Allele refAllele, final Allele altAllele, final String expected) {
        Assert.assertEquals(FuncotatorUtils.getAlternateCodingSequence(refCodingSeq, startPos, refAllele, altAllele), expected);
    }

    @Test(dataProvider = "provideDataForGetEukaryoticAminoAcidByCodon")
    void testGetEukaryoticAminoAcidByCodon(final String codon, final AminoAcid expected) {
        Assert.assertEquals(FuncotatorUtils.getEukaryoticAminoAcidByCodon(codon), expected);
    }

    @Test(dataProvider = "provideDataForGetMitochondrialAminoAcidByCodon")
    void testGetMitochondrialAminoAcidByCodon(final String codon, final boolean isFirst, final AminoAcid expected) {
        Assert.assertEquals(FuncotatorUtils.getMitochondrialAminoAcidByCodon(codon, isFirst), expected);
    }

    @Test
    void testGetAminoAcidNames() {
        Assert.assertEquals(FuncotatorUtils.getAminoAcidNames(),
                new String[]{
                        "Alanine",
                        "Arganine",
                        "Asparagine",
                        "Aspartic acid",
                        "Cysteine",
                        "Glutamic acid",
                        "Glutamine",
                        "Glycine",
                        "Histidine",
                        "Isoleucine",
                        "Leucine",
                        "Lysine",
                        "Methionine",
                        "Phenylalanine",
                        "Proline",
                        "Serine",
                        "Stop codon",
                        "Threonine",
                        "Tryptophan",
                        "Tyrosine",
                        "Valine",
                        "Nonsense Acid"
                }
        );
    }

    @Test
    void testGetAminoAcidCodes() {
        Assert.assertEquals(FuncotatorUtils.getAminoAcidCodes(),
                new String[] {
                        "Ala",
                        "Arg",
                        "Asn",
                        "Asp",
                        "Cys",
                        "Glu",
                        "Gln",
                        "Gly",
                        "His",
                        "Ile",
                        "Leu",
                        "Lys",
                        "Met",
                        "Phe",
                        "Pro",
                        "Ser",
                        "Stop",
                        "Thr",
                        "Trp",
                        "Tyr",
                        "Val",
                         "NONSENSE",
                }
        );
    }

    @Test (dataProvider = "provideDataForIsInFrameWithEndOfRegion")
    void testIsInFrameWithEndOfRegion(final int pos, final int length, final boolean expected) {
        Assert.assertEquals( FuncotatorUtils.isInFrameWithEndOfRegion(pos, length), expected );
    }
}
