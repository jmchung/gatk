package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.variantcontext.VariantContextBuilder;
import htsjdk.variant.variantcontext.VariantContextUtils;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Unit test class for the {@link GencodeFuncotationFactory} class.
 * Created by jonn on 9/1/17.
 */
public class GencodeFuncotationFactoryUnitTest extends BaseTest {

    //==================================================================================================================
    // Static Variables:

    private final String FUNCOTATOR_TEST_DIR = toolsTestDir + "funcotator" + File.separator;

    private final String HG19_CHR19_REFERENCE_FILE_NAME =  "/Users/jonn/Development/references/GRCh37.p13.chr19.fasta";

    private final String MUC16_GENCODE_ANNOTATIONS_FILE_NAME = FUNCOTATOR_TEST_DIR + "MUC16.gencode.v19.annotation.gtf";

    //==================================================================================================================
    // Helper Methods:

    //==================================================================================================================
    // Data Providers:

    @DataProvider
    Object[][] provideSnpDataForGetVariantClassification() {
        final List<Object[]> l = new ArrayList<>();

        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_1() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_2() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_3() );
        
        return l.toArray(new Object[][]{{}});
    }

    //==================================================================================================================
    // Tests:

    @Test (dataProvider = "provideSnpDataForGetVariantClassification")
    void testGetVariantClassification(final int chromosomeNumber,
                                      final int start,
                                      final int end,
                                      final GencodeFuncotation.VariantType variantType,
                                      final String ref,
                                      final String alt,
                                      final GencodeFuncotation.VariantClassification variantClassification) {

        final Allele refAllele = Allele.create(ref, true);
        final Allele altAllele = Allele.create(alt);

        final VariantContextBuilder variantContextBuilder = new VariantContextBuilder(
                HG19_CHR19_REFERENCE_FILE_NAME,
                "chr" + Integer.toString(chromosomeNumber),
                start,
                end,
                Arrays.asList(refAllele, altAllele)
        );
        final VariantContext variantContext = variantContextBuilder.make();

//        getVariantClassification(final VariantContext variant,
//        final Allele altAllele,
//        final GencodeFuncotation.VariantType variantType,
//        final GencodeGtfExonFeature exon,
//        final FuncotatorUtils.SequenceComparison sequenceComparison ){

    }


}
