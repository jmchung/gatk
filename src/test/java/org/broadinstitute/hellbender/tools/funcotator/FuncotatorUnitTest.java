package org.broadinstitute.hellbender.tools.funcotator;

import org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode.GencodeFuncotation;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * Unit tests for the {@link org.broadinstitute.hellbender.tools.funcotator}.
 * Created by jonn on 8/22/17.
 */
public class FuncotatorUnitTest extends BaseTest {

    //==================================================================================================================
    // Helper Methods:

    private static GencodeFuncotation createFuncotation(final String allele, final GencodeFuncotation.VariantClassification classification, final String symbol,
                                                        final String gene, final String featureType, final String feature, final String biotype,
                                                        final int exonNumber, final int cdsPosition, final int proteinPosition ) {

        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        gencodeFuncotation.setAllele(allele);
        gencodeFuncotation.setClassification(classification);
        gencodeFuncotation.setSymbol(symbol);
        gencodeFuncotation.setGene(gene);
        gencodeFuncotation.setFeatureType(featureType);
        gencodeFuncotation.setFeature(feature);
        gencodeFuncotation.setBiotype(biotype);
        gencodeFuncotation.setExon(exonNumber);
        gencodeFuncotation.setCdsPosition(cdsPosition);
        gencodeFuncotation.setProteinPosition(proteinPosition);

        return gencodeFuncotation;
    }

    //==================================================================================================================
    // Data Providers:

    @DataProvider
    Object[][] createFuncotationsAndStringSerializations() {

        final String D = GencodeFuncotation.FIELD_DELIMITER;

        return new Object[][] {
                {
                        createFuncotation(null, GencodeFuncotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                        "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        ""
                },
                {
                        createFuncotation("A", null, "ABC1", "Alphabet Gene",
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", GencodeFuncotation.VariantClassification.START_CODON_SNP, null, "Alphabet Gene",
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", GencodeFuncotation.VariantClassification.START_CODON_SNP, "ABC1", null,
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", GencodeFuncotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                null, "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", GencodeFuncotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                "Words", null, "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", GencodeFuncotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                "Words", "NumbersAndLetters", null, 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", GencodeFuncotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
        };
    }


    //==================================================================================================================
    // Tests:

    @Test(dataProvider = "createFuncotationsAndStringSerializations")
    void testFuncotationSerializeToVcfString(final GencodeFuncotation gencodeFuncotation, final String expected) {
        Assert.assertEquals(gencodeFuncotation.serializeToVcfString(), expected);
    }


}
