package org.broadinstitute.hellbender.tools.funcotator;

import htsjdk.samtools.util.Locatable;
import htsjdk.variant.variantcontext.Allele;
import lombok.Data;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.ReferenceFileSource;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.File;
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
     * 012345678901234567890
     * TCATCTGCAGGTGTCTGACTT
     *
     */
    private void printReferenceBases() {
        final ReferenceContext ref = new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, 12000, 16000));

        // Ones place:
        final StringBuilder sb_o = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length ; i+=10 ) {
            sb_o.append("0123456789");
        }
        // Tens place:
        final StringBuilder sb_t = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length ; ++i ) {
            sb_t.append((int)(i / 10.0) % 10);
        }
        // Hundreds place:
        final StringBuilder sb_h = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length ; ++i ) {
            sb_h.append((int)(i / 100.0) % 10);
        }
        // Thousands place:
        final StringBuilder sb_th = new StringBuilder();
        for( int i = 0 ; i < ref.getBases().length ; ++i ) {
            sb_th.append((int)(i / 1000.0) % 10);
        }

        System.out.println();
        System.out.println("Location: " + TEST_REFERENCE_CONTIG + ":" + TEST_REFERENCE_START + ":" + TEST_REFERENCE_END);
        System.out.println("=================================================================================");
        System.out.println( sb_th.toString() );
        System.out.println( sb_h.toString() );
        System.out.println( sb_t.toString() );
        System.out.println( sb_o.toString() );
        System.out.println( new String(ref.getBases()) );
    }

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
                        "CAGAGACGGGAGGGGCAGAGCCGCAGGCACAGCCAAGAGGGCTGAAGAAAT"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Arrays.asList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 550),
                                new SimpleInterval("1", TEST_REFERENCE_START + 551, TEST_REFERENCE_START + 600)
                        ),
                        "CAGAGACGGGAGGGGCAGAGCCGCAGGCACAGCCAAGAGGGCTGAAGAAATGGTAGAACGGAGCAGCTGGTGATGTGTGGGCCCACCGGCCCCAGGCTCCT"
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
                        "CAGAGACGGG"
                },
                {
                        new ReferenceContext(new ReferenceFileSource(TEST_REFERENCE), new SimpleInterval(TEST_REFERENCE_CONTIG, TEST_REFERENCE_START, TEST_REFERENCE_END)),
                        Collections.singletonList(
                                new SimpleInterval("1", TEST_REFERENCE_START + 500, TEST_REFERENCE_START + 500)
                        ),
                        "C"
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
        return new Object[][] {
                {
                    new SimpleInterval("chr1", 0, 1),
                    Arrays.asList(
                            new SimpleInterval("chr1", 10,19),
                            new SimpleInterval("chr1", 30,39),
                            new SimpleInterval("chr1", 50,59),
                            new SimpleInterval("chr1", 70,79),
                            new SimpleInterval("chr1", 90,99)
                    ),

                },
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
                    "01234567890A1234567890123456789", 11, Allele.create((byte)'A'), Allele.create((byte)'A'), "01234567890A1234567890123456789"
                },
                {
                    "01234567890A1234567890123456789", 11, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "01234567890ATGCATGC1234567890123456789"
                },
                {
                    "A", 0, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "ATGCATGC"
                },
                {
                    "BA", 1, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "BATGCATGC"
                },
                {
                    "AB", 0, Allele.create((byte)'A'), Allele.create("ATGCATGC".getBytes()), "ATGCATGCB"
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
}
