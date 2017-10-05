package org.broadinstitute.hellbender.tools.funcotator;

import htsjdk.samtools.util.Locatable;
import htsjdk.tribble.annotation.Strand;
import htsjdk.variant.variantcontext.Allele;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.ReferenceFileSource;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.read.ReadUtils;
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
                        Strand.POSITIVE,
                        ""
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)),
                        Strand.POSITIVE,
                        "GCAGAGACGGGAGGGGCAGAGCCGCAGGCACAGCCAAGAGGGCTGAAGAAA"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Arrays.asList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550),
                                new SimpleInterval("1", TEST_REFERENCE_START + 551, TEST_REFERENCE_START + 600)
                        ),
                        Strand.POSITIVE,
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
                        Strand.POSITIVE,
                        "GCAGAGACGG"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500)
                        ),
                        Strand.POSITIVE,
                        "G"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, 1, 10)),
                        Collections.singletonList(
                                new SimpleInterval("1", 1, 1)
                        ),
                        Strand.POSITIVE,
                        "N"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.emptyList(),
                        Strand.NEGATIVE,
                        ""
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)),
                        Collections.singletonList(new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)),
                        Strand.NEGATIVE,
                        "TTTCTTCAGCCCTCTTGGCTGTGCCTGCGGCTCTGCCCCTCCCGTCTCTGC"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 600)),
                        Arrays.asList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550),
                                new SimpleInterval("1", TEST_REFERENCE_START + 551, TEST_REFERENCE_START + 600)
                        ),
                        Strand.NEGATIVE,
                        "GGAGCCTGGGGCCGGTGGGCCCACACATCACCAGCTGCTCCGTTCTACCATTTCTTCAGCCCTCTTGGCTGTGCCTGCGGCTCTGCCCCTCCCGTCTCTGC"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Arrays.asList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 509, TEST_REFERENCE_START + 509),
                                new SimpleInterval("1", TEST_REFERENCE_START + 508, TEST_REFERENCE_START + 508),
                                new SimpleInterval("1", TEST_REFERENCE_START + 507, TEST_REFERENCE_START + 507),
                                new SimpleInterval("1", TEST_REFERENCE_START + 506, TEST_REFERENCE_START + 506),
                                new SimpleInterval("1", TEST_REFERENCE_START + 505, TEST_REFERENCE_START + 505),
                                new SimpleInterval("1", TEST_REFERENCE_START + 504, TEST_REFERENCE_START + 504),
                                new SimpleInterval("1", TEST_REFERENCE_START + 503, TEST_REFERENCE_START + 503),
                                new SimpleInterval("1", TEST_REFERENCE_START + 502, TEST_REFERENCE_START + 502),
                                new SimpleInterval("1", TEST_REFERENCE_START + 501, TEST_REFERENCE_START + 501),
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500)
                        ),
                        Strand.NEGATIVE,
                        "TGGTCAGCCACTGCAGCC"
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
                        Strand.NEGATIVE,
                        "TGGTCAGCCACTGCAGCC"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START + 500, TEST_REFERENCE_END + 500)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500)
                        ),
                        Strand.NEGATIVE,
                        "C"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500)
                        ),
                        Strand.NEGATIVE,
                        "T"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START, TEST_REFERENCE_START)
                        ),
                        Strand.NEGATIVE,
                        "C"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, 1, 10)),
                        Collections.singletonList(
                                new SimpleInterval("1", 1, 1)
                        ),
                        Strand.NEGATIVE,
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
                        Strand.POSITIVE
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("2", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)
                        ),
                        Strand.NEGATIVE
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("2", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550)
                        ),
                        Strand.NONE
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

        final List<? extends Locatable> exons_forward = Arrays.asList(
                new SimpleInterval("chr1", 10,19),
                new SimpleInterval("chr1", 30,39),
                new SimpleInterval("chr1", 50,59),
                new SimpleInterval("chr1", 70,79),
                new SimpleInterval("chr1", 90,99)
        );

        final List<? extends Locatable> exons_backward = Arrays.asList(
                new SimpleInterval("chr1", 90,99),
                new SimpleInterval("chr1", 70,79),
                new SimpleInterval("chr1", 50,59),
                new SimpleInterval("chr1", 30,39),
                new SimpleInterval("chr1", 10,19)
        );

        return new Object[][] {
                { new SimpleInterval("chr1", 1, 1),     exons_forward, Strand.POSITIVE, -1 },
                { new SimpleInterval("chr1", 25, 67),   exons_forward, Strand.POSITIVE, -1 },
                { new SimpleInterval("chr1", 105, 392), exons_forward, Strand.POSITIVE, -1 },
                { new SimpleInterval("chr1", 10, 10),   exons_forward, Strand.POSITIVE,  1 },
                { new SimpleInterval("chr1", 99, 99),   exons_forward, Strand.POSITIVE, 50 },
                { new SimpleInterval("chr1", 50, 67),   exons_forward, Strand.POSITIVE, 21 },

                { new SimpleInterval("chr1", 1, 1),     exons_backward, Strand.NEGATIVE, -1 },
                { new SimpleInterval("chr1", 25, 67),   exons_backward, Strand.NEGATIVE, -1 },
                { new SimpleInterval("chr1", 105, 392), exons_backward, Strand.NEGATIVE, -1 },
                { new SimpleInterval("chr1", 10, 10),   exons_backward, Strand.NEGATIVE, 50 },
                { new SimpleInterval("chr1", 99, 99),   exons_backward, Strand.NEGATIVE,  1 },
                { new SimpleInterval("chr1", 50, 67),   exons_backward, Strand.NEGATIVE, 30 },
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
    Object[][] providePositionsAndFrameshiftResults() {
        return new Object[][] {
                { 1,1,1, false },
                { 1,3,1, true },
                { 1,3,2, true },
                { 1,3,3, false },
                { 1,3,233, true },
                { 1,3,234, false },
                { 1,3,235, true },
                { 8,9,8, true },
                { 8,9,9, false },
                { 8,9,10, true },
                { 8,9,11, true },
                { 8,9,12, false },
        };
    }

    @DataProvider
    Object[][] providePositionAndExpectedAlignedPosition() {
        return new Object[][] {
                {1,1},
                {2,1},
                {3,1},
                {4,4},
                {5,4},
                {6,4},
                {1635,1633},
                {1636,1636},
                {1637,1636},
        };
    }

    @DataProvider
    Object[][] providePositionAndExpectedAlignedEndPosition() {
        return new Object[][] {
                {1,1,3},
                {1,2,3},
                {1,3,3},
                {1,4,6},
                {1,5,6},
                {1,6,6},
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
                {
                    "ATGCATGC", 2, Allele.create((byte)'T'), Allele.create((byte)'G'), "AGGCATGC"
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
    
    @DataProvider
    Object[][] provideStringDataForGetAminoAcidByLetter() {
        return new Object[][] {
                { "A", AminoAcid.ALANINE },
                { "R", AminoAcid.ARGANINE },
                { "N", AminoAcid.ASPARAGINE },
                { "D", AminoAcid.ASPARTIC_ACID },
                { "C", AminoAcid.CYSTEINE },
                { "E", AminoAcid.GLUTAMIC_ACID },
                { "Q", AminoAcid.GLUTAMINE },
                { "G", AminoAcid.GLYCINE },
                { "H", AminoAcid.HISTIDINE },
                { "I", AminoAcid.ISOLEUCINE },
                { "L", AminoAcid.LEUCINE },
                { "K", AminoAcid.LYSINE },
                { "M", AminoAcid.METHIONINE },
                { "F", AminoAcid.PHENYLALANINE },
                { "P", AminoAcid.PROLINE },
                { "S", AminoAcid.SERINE },
                { "*", AminoAcid.STOP_CODON },
                { "T", AminoAcid.THREONINE },
                { "W", AminoAcid.TRYPTOPHAN },
                { "Y", AminoAcid.TYROSINE },
                { "V", AminoAcid.VALINE },
                { "!", AminoAcid.NONSENSE },
                { "ahuewifaef", null },
                { "", null },
                { "X", null },
                { "x", null },
                { "7", null },
        };
    }

    @DataProvider
    Object[][] provideCharDataForGetAminoAcidByLetter() {
        return new Object[][] {
                { 'A', AminoAcid.ALANINE },
                { 'R', AminoAcid.ARGANINE },
                { 'N', AminoAcid.ASPARAGINE },
                { 'D', AminoAcid.ASPARTIC_ACID },
                { 'C', AminoAcid.CYSTEINE },
                { 'E', AminoAcid.GLUTAMIC_ACID },
                { 'Q', AminoAcid.GLUTAMINE },
                { 'G', AminoAcid.GLYCINE },
                { 'H', AminoAcid.HISTIDINE },
                { 'I', AminoAcid.ISOLEUCINE },
                { 'L', AminoAcid.LEUCINE },
                { 'K', AminoAcid.LYSINE },
                { 'M', AminoAcid.METHIONINE },
                { 'F', AminoAcid.PHENYLALANINE },
                { 'P', AminoAcid.PROLINE },
                { 'S', AminoAcid.SERINE },
                { '*', AminoAcid.STOP_CODON },
                { 'T', AminoAcid.THREONINE },
                { 'W', AminoAcid.TRYPTOPHAN },
                { 'Y', AminoAcid.TYROSINE },
                { 'V', AminoAcid.VALINE },
                { '!', AminoAcid.NONSENSE },
                { 'a', null },
                { '\0', null },
                { 'X', null },
                { 'x', null },
                { '7', null },
        };
    }

    @DataProvider
    Object[][] provideDataForGetProteinChangePosition() {
        return new Object[][] {
                { 1, 1 },
                { 2, 1 },
                { 3, 1 },
                { 4, 2 },
                { 5, 2 },
                { 6, 2 },
                { 300, 100 },
                { 301, 101 },
                { 302, 101 },
                { 303, 101 },
                { 304, 102 },
        };
    }

    @DataProvider
    Object[][] provideDataForGetProteinChangeEndPosition() {
        return new Object[][] {
                {1,  3, 1},
                {1,  6, 2},
                {1,  9, 3},
                {1, 12, 4},
                {1, 15, 5},
                {1, 18, 6},
                {1, 21, 7},
                {1, 24, 8},
        };
    }

    @DataProvider
    Object[][] provideDataForGetAlignedAllele() {

        final String seq = "ATGAAAGGGGTGCCTATGCTAGATAGACAGATAGTGTGTGTGTGTGTGCGCGCGCGCGCGCGTTGTTAG";

        //CTA ACA ACG CGC GCG CGC GCG CAC ACA CAC ACA CAC TAT CTG TCT ATC TAG CAT AGG CAC CCC TTT CAT

        return new Object[][] {
                { seq,  1, 3,  Strand.POSITIVE, "ATG" },
                { seq,  4, 6,  Strand.POSITIVE, "AAA" },
                { seq,  7, 9,  Strand.POSITIVE, "GGG" },
                { seq, 10, 12, Strand.POSITIVE, "GTG" },
                { seq, 13, 15, Strand.POSITIVE, "CCT" },
                { seq, 16, 18, Strand.POSITIVE, "ATG" },
                { seq, 19, 21, Strand.POSITIVE, "CTA" },
                { seq,  1,  6, Strand.POSITIVE, "ATGAAA" },
                { seq,  4,  9, Strand.POSITIVE, "AAAGGG" },
                { seq,  7, 12, Strand.POSITIVE, "GGGGTG" },
                { seq, 10, 15, Strand.POSITIVE, "GTGCCT" },
                { seq, 13, 18, Strand.POSITIVE, "CCTATG" },
                { seq, 16, 21, Strand.POSITIVE, "ATGCTA" },
                { seq, 19, 24, Strand.POSITIVE, "CTAGAT" },
                { seq, 1, seq.length(), Strand.POSITIVE, seq },

                { seq,  1, 3,  Strand.NEGATIVE, "CTA" },
                { seq,  4, 6,  Strand.NEGATIVE, "ACA" },
                { seq,  7, 9,  Strand.NEGATIVE, "ACG" },
                { seq, 10, 12, Strand.NEGATIVE, "CGC" },
                { seq, 13, 15, Strand.NEGATIVE, "GCG" },
                { seq, 16, 18, Strand.NEGATIVE, "CGC" },
                { seq, 19, 21, Strand.NEGATIVE, "GCG" },
                { seq,  1,  6, Strand.NEGATIVE, "CTAACA" },
                { seq,  4,  9, Strand.NEGATIVE, "ACAACG" },
                { seq,  7, 12, Strand.NEGATIVE, "ACGCGC" },
                { seq, 10, 15, Strand.NEGATIVE, "CGCGCG" },
                { seq, 13, 18, Strand.NEGATIVE, "GCGCGC" },
                { seq, 16, 21, Strand.NEGATIVE, "CGCGCG" },
                { seq, 19, 24, Strand.NEGATIVE, "GCGCAC" },
                { seq, 1, seq.length(), Strand.NEGATIVE, ReadUtils.getBasesReverseComplement( seq.getBytes() ) },
        };
    }

    @DataProvider
    Object[][] provideDataForGetCodingSequenceAlleleStartPosition() {

        return new Object[][] {
                { 1,  1, 10, Strand.POSITIVE,  1},
                { 5,  1, 10, Strand.POSITIVE,  5},
                { 10, 1, 10, Strand.POSITIVE, 10},

                { 1,  1, 10, Strand.NEGATIVE, 10},
                { 5,  1, 10, Strand.NEGATIVE,  6},
                { 10, 1, 10, Strand.NEGATIVE,  1},
        };
    }

    @DataProvider
    Object[][] provideDataForTestCreateSpliceSiteCodonChange() {

        return new Object[][] {
                {1000, 5, 1000, 1500, Strand.POSITIVE, "c.e5-0"},
                {1000, 4, 0, 1500, Strand.POSITIVE,    "c.e4+500"},
                {1000, 3, 500, 1500, Strand.POSITIVE,  "c.e3-500"},

                {1000, 5, 1000, 1500, Strand.NEGATIVE, "c.e5+0"},
                {1000, 4, 0, 1500, Strand.NEGATIVE,    "c.e4-500"},
                {1000, 3, 500, 1500, Strand.NEGATIVE,  "c.e3+500"},

                {1000, 5, 1500, 500, Strand.NEGATIVE,  "c.e5+500"},
        };
    }

    //==================================================================================================================
    // Tests:

    @Test(dataProvider = "provideAllelesAndFrameshiftResults")
    void testIsFrameshift(final Allele ref, final Allele alt, final boolean expected) {
        Assert.assertEquals( FuncotatorUtils.isFrameshift(ref, alt), expected );
    }

    @Test(dataProvider = "providePositionsAndFrameshiftResults")
    void testIsFrameshiftByPositions(final int refStart, final int refEnd, final int altEnd, final boolean expected) {
        Assert.assertEquals( FuncotatorUtils.isFrameshift(refStart, refEnd, altEnd), expected );
    }

//    @Test(dataProvider = "provideReferenceAndExonListAndExpected")
//    void testGetCodingSequence(final ReferenceContext reference, final List<Locatable> exonList, final Strand strand, final String expected) {
//        final String codingSequence = FuncotatorUtils.getCodingSequence(reference, exonList, strand);
//        Assert.assertEquals( codingSequence, expected );
//    }

//    @Test(dataProvider = "provideReferenceAndExonListForGatkExceptions",
//            expectedExceptions = GATKException.class)
//    void testGetCodingSequenceWithGatkExceptions(final ReferenceContext reference, final Strand strand, final List<Locatable> exonList) {
//        FuncotatorUtils.getCodingSequence(reference, exonList, strand);
//    }

//    @Test(dataProvider = "provideReferenceAndExonListForIllegalArgumentExceptions",
//            expectedExceptions = IllegalArgumentException.class)
//    void testGetCodingSequenceWithIllegalArgumentExceptions(final ReferenceContext reference, final List<Locatable> exonList) {
//        FuncotatorUtils.getCodingSequence(reference, exonList);
//    }

    @Test(dataProvider = "provideDataForGetStartPositionInTranscript")
    void testGetStartPositionInTranscript(final Locatable variant, final List<? extends Locatable> transcript, final Strand strand, final int expected) {
        Assert.assertEquals( FuncotatorUtils.getStartPositionInTranscript(variant, transcript, strand), expected );
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

    @Test (dataProvider = "provideStringDataForGetAminoAcidByLetter")
    void testGetAminoAcidByLetter(final String letter, final AminoAcid expected) {
        Assert.assertEquals( FuncotatorUtils.getAminoAcidByLetter(letter), expected );
    }

    @Test (dataProvider = "provideCharDataForGetAminoAcidByLetter")
    void testGetAminoAcidByLetter(final char letter, final AminoAcid expected) {
        Assert.assertEquals( FuncotatorUtils.getAminoAcidByLetter(letter), expected );
    }

    @Test (dataProvider = "provideDataForGetProteinChangePosition")
    void testGetProteinChangePosition(final Integer alignedCodingSequenceStartPos, final int expected) {
        Assert.assertEquals( FuncotatorUtils.getProteinChangePosition(alignedCodingSequenceStartPos) , expected );
    }

    @Test (dataProvider = "provideDataForGetProteinChangeEndPosition")
    void testGetProteinChangeEndPosition(final Integer proteinChangeStartPosition, final Integer alignedAlternateAlleleLength, final int expected) {
        Assert.assertEquals( FuncotatorUtils.getProteinChangeEndPosition(proteinChangeStartPosition, alignedAlternateAlleleLength) , expected );
    }

    @Test (dataProvider = "provideDataForGetAlignedAllele")
    void testGetAlignedAllele(  final String refSequence,
                                final Integer alignedAlleleStart,
                                final Integer alignedAlleleStop,
                                final Strand strand,
                                final String expected) {
        Assert.assertEquals( FuncotatorUtils.getAlignedAllele(refSequence, alignedAlleleStart, alignedAlleleStop, strand), expected );
    }

    @Test (dataProvider = "provideDataForGetCodingSequenceAlleleStartPosition")
    void testGetCodingSequenceAlleleStartPosition(final int variantStartPosition, final int codingStartPosition, final int codingEndPosition, final Strand strand, final int expected) {
        Assert.assertEquals( FuncotatorUtils.getCodingSequenceAlleleStartPosition(variantStartPosition, codingStartPosition, codingEndPosition, strand), expected );
    }

    @Test (dataProvider = "provideDataForTestCreateSpliceSiteCodonChange")
    void testCreateSpliceSiteCodonChange(final int variantStart,
                                         final int exonNumber,
                                         final int exonStart,
                                         final int exonEnd,
                                         final Strand strand,
                                         final String expected) {

        Assert.assertEquals( FuncotatorUtils.createSpliceSiteCodonChange(variantStart, exonNumber, exonStart, exonEnd, strand), expected );
    }

}
