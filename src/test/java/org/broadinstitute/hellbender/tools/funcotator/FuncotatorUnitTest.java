package org.broadinstitute.hellbender.tools.funcotator;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * Unit tests for the {@link org.broadinstitute.hellbender.tools.funcotator}.
 * Created by jonn on 8/22/17.
 */
public class FuncotatorUnitTest {

    //==================================================================================================================
    // Helper Methods:

    private static Funcotation createFuncotation(final String allele, final Funcotation.VariantClassification classification, final String symbol,
                                                 final String gene, final String featureType, final String feature, final String biotype,
                                                 final int exonNumber, final int cdsPosition, final int proteinPosition ) {

        final Funcotation funcotation = new Funcotation();

        funcotation.setAllele(allele);
        funcotation.setClassification(classification);
        funcotation.setSymbol(symbol);
        funcotation.setGene(gene);
        funcotation.setFeatureType(featureType);
        funcotation.setFeature(feature);
        funcotation.setBiotype(biotype);
        funcotation.setExon(exonNumber);
        funcotation.setCdsPosition(cdsPosition);
        funcotation.setProteinPosition(proteinPosition);

        return funcotation;
    }

    //==================================================================================================================
    // Data Providers:

    @DataProvider
    Object[][] createFuncotationsAndStringSerializations() {

        final String D = Funcotation.FIELD_DELIMITER;

        return new Object[][] {
                {
                        createFuncotation(null, Funcotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
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
                        createFuncotation("A", Funcotation.VariantClassification.START_CODON_SNP, null, "Alphabet Gene",
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", Funcotation.VariantClassification.START_CODON_SNP, "ABC1", null,
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", Funcotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                null, "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", Funcotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                "Words", null, "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", Funcotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                "Words", "NumbersAndLetters", null, 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + D + "1" + D + "2" + D + "3"
                },
                {
                        createFuncotation("A", Funcotation.VariantClassification.START_CODON_SNP, "ABC1", "Alphabet Gene",
                                "Words", "NumbersAndLetters", "Cookie Monster", 1, 2, 3),

                        "A" + D + "START_CODON_SNP" + D + "ABC1" + D + "Alphabet Gene" + D + "Words" + D +
                                "NumbersAndLetters" + D + "Cookie Monster" + D + "1" + D + "2" + D + "3"
                },
        };
    }


    //==================================================================================================================
    // Tests:

    @Test(dataProvider = "createFuncotationsAndStringSerializations")
    void testFuncotationSerializeToVcfString(final Funcotation funcotation, final String expected) {
        Assert.assertEquals(funcotation.serializeToVcfString(), expected);
    }


}
