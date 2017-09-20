package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import htsjdk.samtools.util.Locatable;
import htsjdk.tribble.AbstractFeatureReader;
import htsjdk.tribble.CloseableTribbleIterator;
import htsjdk.tribble.FeatureReader;
import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.variantcontext.VariantContextBuilder;
import org.broadinstitute.hellbender.engine.*;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Unit test class for the {@link GencodeFuncotationFactory} class.
 * Created by jonn on 9/1/17.
 */
public class GencodeFuncotationFactoryUnitTest extends BaseTest {

    //==================================================================================================================
    // Static Variables:

    private static final String FUNCOTATOR_TEST_DIR = toolsTestDir + "funcotator" + File.separator;
    private static final String HG19_CHR19_REFERENCE_FILE_NAME = "/Users/jonn/Development/references/GRCh37.p13.chr19.fasta";
    private static final String MUC16_GENCODE_ANNOTATIONS_FILE_NAME = FUNCOTATOR_TEST_DIR + "gencode.v19.MUC16.gtf";
    private static final String MUC16_GENCODE_TRANSCRIPT_FASTA_FILE = FUNCOTATOR_TEST_DIR + "gencode.v19.MUC16_transcript.fasta";
    private static final String MUC_16_TRANSCRIPT = "ENST00000397910.4";

    private static final FeatureReader<GencodeGtfFeature> muc16FeatureReader;
    private static final ReferenceDataSource refDataSource;

    // Initialization of static variables:
    static {
        muc16FeatureReader = AbstractFeatureReader.getFeatureReader( MUC16_GENCODE_ANNOTATIONS_FILE_NAME, new GencodeGtfCodec() );
        refDataSource = ReferenceDataSource.of( new File (HG19_CHR19_REFERENCE_FILE_NAME) );
    }

    //==================================================================================================================
    // Helper Methods:

    private static GencodeGtfExonFeature getExonForVariant( final CloseableTribbleIterator<GencodeGtfFeature> gtfFeatureIterator,
                                                            final SimpleInterval variantLocation ) {

        final Optional<GencodeGtfExonFeature> exonOption = gtfFeatureIterator.stream()
                                                    .map( f -> ((GencodeGtfGeneFeature) f))
                                                    .flatMap(g -> g.getTranscripts().stream())
                                                    .flatMap(t -> t.getExons().stream())
                                                    .filter( x -> x.overlaps(variantLocation) )
                                                    .findFirst();

        return exonOption.orElseThrow( () -> new GATKException("Could not get an exon associated with this variant's interval: " + variantLocation.toString()) );
    }

    private static GencodeGtfExonFeature getExonForVariant( final GencodeGtfGeneFeature gene,
                                                            final SimpleInterval variantLocation ) {

        final Optional<GencodeGtfExonFeature> exonOption = gene.getTranscripts().stream()
                                                                .flatMap(t -> t.getExons().stream())
                                                                .filter( x -> x.overlaps(variantLocation) )
                                                                .findFirst();

        return exonOption.orElseThrow( () -> new GATKException("Could not get an exon associated with this variant's interval: " + variantLocation.toString()) );
    }

    private static GencodeGtfTranscriptFeature getMuc16Transcript(final GencodeGtfGeneFeature gene) {
        final Optional<GencodeGtfTranscriptFeature> transcriptOption = gene.getTranscripts().stream()
                .filter( x -> x.getTranscriptId().equals(MUC_16_TRANSCRIPT) )
                .findFirst();

        return transcriptOption.orElseThrow( () -> new GATKException("Could not get the MUC16 transcript from the MUC16 gene!  The test has mutated!  No good!") );
    }

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

    @DataProvider
    Object[][] provideDataForCreateFuncotations() {
        return new Object[][]{
                {},
        };
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
                                      final GencodeFuncotation.VariantClassification expectedVariantClassification) {

        final String contig = "chr" + Integer.toString(chromosomeNumber);
        final SimpleInterval variantInterval = new SimpleInterval( contig, start, end );

        final Allele refAllele = Allele.create(ref, true);
        final Allele altAllele = Allele.create(alt);

        final VariantContextBuilder variantContextBuilder = new VariantContextBuilder(
                HG19_CHR19_REFERENCE_FILE_NAME,
                contig,
                start,
                end,
                Arrays.asList(refAllele, altAllele)
        );
        final VariantContext variantContext = variantContextBuilder.make();

        try {
            // Get our gene feature iterator:
            final CloseableTribbleIterator<GencodeGtfFeature> gtfFeatureIterator = muc16FeatureReader.query(contig, start, end);

            // Get the gene.
            // We know the first gene is the right one - the gene in question is the MUC16 gene:
            final GencodeGtfGeneFeature             gene = (GencodeGtfGeneFeature) gtfFeatureIterator.next();
            final GencodeGtfTranscriptFeature transcript = getMuc16Transcript(gene);
            final GencodeGtfExonFeature             exon = getExonForVariant( gene, variantInterval );

            final ReferenceContext referenceContext = new ReferenceContext( refDataSource, variantInterval );

            final List<? extends Locatable> exonPositionList = GencodeFuncotationFactory.getSortedExonPositions(transcript);

            final ReferenceDataSource muc16TranscriptDataSource = ReferenceDataSource.of(new File(MUC16_GENCODE_TRANSCRIPT_FASTA_FILE));
            final Map<String, GencodeFuncotationFactory.MappedTranscriptIdInfo> muc16TranscriptIdMap = GencodeFuncotationFactory.createTranscriptIdMap(muc16TranscriptDataSource);

            final FuncotatorUtils.SequenceComparison seqComp =
                    GencodeFuncotationFactory.createSequenceComparison(
                            variantContext,
                            altAllele,
                            referenceContext,
                            transcript,
                            exonPositionList,
                            muc16TranscriptIdMap,
                            muc16TranscriptDataSource);

            final GencodeFuncotation.VariantClassification varClass = GencodeFuncotationFactory.getVariantClassification(
                    variantContext,
                    altAllele,
                    variantType,
                    exon,
                    seqComp
            );

            Assert.assertEquals( varClass, expectedVariantClassification );

        }
        catch ( final IOException ex ) {
            throw new GATKException("Could not finish the test!", ex);
        }
    }

    @Test (dataProvider = "provideDataForCreateFuncotations")
    void testCreateFuncotations() {
//    private List<GencodeFuncotation> createFuncotations(final VariantContext variant,
//                                                        final Allele altAllele,
//                                                        final GencodeGtfGeneFeature gtfFeature,
//                                                        final ReferenceContext reference);
    }

}
