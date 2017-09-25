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
    Object[][] provideMuc16SnpDataForGetVariantClassification() {
        final List<Object[]> l = new ArrayList<>();

        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_1() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_2() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_3() );
        
        return l.toArray(new Object[][]{{}});
    }

    @DataProvider
    Object[][] provideMuc16SnpDataForGetVariantClassificationWithOutOfCdsData() {
        final List<Object[]> l = new ArrayList<>();

        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_1() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_2() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_3() );
        l.addAll( DataProviderForSnpsOnMuc16.provideSnpDataForGetVariantClassification_NotInCDS() );

        return l.toArray(new Object[][]{{}});
    }
    
    @DataProvider
    Object[][] provideMuc16MnpDataForCreateFuncotations() {
        final List<Object[]> l = new ArrayList<>();

        l.addAll( DataProviderForMuc16MnpFullData.provideMnpDataForMuc16_1() );
        l.addAll( DataProviderForMuc16MnpFullData.provideMnpDataForMuc16_2() );
        l.addAll( DataProviderForMuc16MnpFullData.provideMnpDataForMuc16_3() );
        l.addAll( DataProviderForMuc16MnpFullData.provideMnpDataForMuc16_4() );
        l.addAll( DataProviderForMuc16MnpFullData.provideMnpDataForMuc16_5() );

        return l.toArray(new Object[][]{{}});
    }

    //==================================================================================================================
    // Tests:

    @Test (dataProvider = "provideMuc16SnpDataForGetVariantClassification")
    void testGetVariantClassificationForCodingRegions(final int chromosomeNumber,
                                      final int start,
                                      final int end,
                                      final GencodeFuncotation.VariantType variantType,
                                      final String ref,
                                      final String alt,
                                      final GencodeFuncotation.VariantClassification expectedVariantClassification) {

        // This test can only deal with variants in coding regions.
        // So we ignore any tests that are expected outside of coding regions.
        // i.e. expectedVariantClassification is one of:
        //     { INTRON, FIVE_PRIME_UTR, THREE_PRIME_UTR, IGR, FIVE_PRIME_FLANK, DE_NOVO_START_IN_FRAME, DE_NOVO_START_OUT_FRAME, RNA, LINCRNA }
        // We test these cases in another unit test.
        if ((expectedVariantClassification == GencodeFuncotation.VariantClassification.INTRON) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.FIVE_PRIME_UTR) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.THREE_PRIME_UTR) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.IGR) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.FIVE_PRIME_FLANK) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.DE_NOVO_START_IN_FRAME) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.DE_NOVO_START_OUT_FRAME) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.RNA) ||
            (expectedVariantClassification == GencodeFuncotation.VariantClassification.LINCRNA) )
        {
            return;
        }

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

        // Get our gene feature iterator:
        final CloseableTribbleIterator<GencodeGtfFeature> gtfFeatureIterator;
        try {
            gtfFeatureIterator = muc16FeatureReader.query(contig, start, end);
        }
        catch (final IOException ex) {
            throw new GATKException("Could not finish the test!", ex);
        }

        // Get the gene.
        // We know the first gene is the right one - the gene in question is the MUC16 gene:
        final GencodeGtfGeneFeature             gene = (GencodeGtfGeneFeature) gtfFeatureIterator.next();
        final GencodeGtfTranscriptFeature transcript = getMuc16Transcript(gene);
        final GencodeGtfExonFeature             exon = getExonForVariant( gene, variantInterval );

        final ReferenceContext referenceContext = new ReferenceContext( refDataSource, variantInterval );

        final List<? extends Locatable> exonPositionList = GencodeFuncotationFactory.getSortedExonAndStartStopPositions(transcript);

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

    @Test (dataProvider = "provideMuc16SnpDataForGetVariantClassificationWithOutOfCdsData")
    void testCreateFuncotationsMuc16Snp(final int chromosomeNumber,
                                final int start,
                                final int end,
                                final GencodeFuncotation.VariantType expectedVariantType,
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

        // Get our gene feature iterator:
        final CloseableTribbleIterator<GencodeGtfFeature> gtfFeatureIterator;
        try {
            gtfFeatureIterator = muc16FeatureReader.query(contig, start, end);
        }
        catch (final IOException ex) {
            throw new GATKException("Could not finish the test!", ex);
        }

        // Get the gene.
        // We know the first gene is the right one - the gene in question is the MUC16 gene:
        final GencodeGtfGeneFeature gene = (GencodeGtfGeneFeature) gtfFeatureIterator.next();
        final ReferenceContext referenceContext = new ReferenceContext( refDataSource, variantInterval );

        // Create a factory for our funcotations:
        try (final GencodeFuncotationFactory funcotationFactory = new GencodeFuncotationFactory(new File(MUC16_GENCODE_TRANSCRIPT_FASTA_FILE))) {
            // Generate our funcotations:
            final List<GencodeFuncotation> funcotations = funcotationFactory.createFuncotations(variantContext, altAllele, gene, referenceContext);

            // Make sure we get what we expected:
            Assert.assertEquals(funcotations.size(), 1);
            Assert.assertEquals(funcotations.get(0).getVariantClassification(), expectedVariantClassification);
            Assert.assertEquals(funcotations.get(0).getVariantType(), expectedVariantType);
        }
    }

    @Test (dataProvider = "provideMuc16MnpDataForCreateFuncotations")
    //new Object[]{"MUC16", 19, 9091811, 9091811, "Silent", "SNP", "G", "A", "g.chr19:9091811G>A", "-", "c.4C>T", "c.(4-6)Ctg>Ttg", "p.L2L"},
    void testCreateFuncotationsMuc16Mnp(final String expectedGeneName,
                                final int chromosomeNumber,
                                final int start,
                                final int end,
                                final GencodeFuncotation.VariantClassification expectedVariantClassification,
                                final GencodeFuncotation.VariantType expectedVariantType,
                                final String ref,
                                final String alt,
                                final String expectedGenomeChange,
                                final String expectedStrand,
                                final String expectedCDnaChange,
                                final String expectedCodonChange,
                                final String expectedProteinChange) {


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

        // Get our gene feature iterator:
        final CloseableTribbleIterator<GencodeGtfFeature> gtfFeatureIterator;
        try {
            gtfFeatureIterator = muc16FeatureReader.query(contig, start, end);
        }
        catch (final IOException ex) {
            throw new GATKException("Could not finish the test!", ex);
        }

        // Get the gene.
        // We know the first gene is the right one - the gene in question is the MUC16 gene:
        final GencodeGtfGeneFeature gene = (GencodeGtfGeneFeature) gtfFeatureIterator.next();
        final ReferenceContext referenceContext = new ReferenceContext( refDataSource, variantInterval );

        // Create a factory for our funcotations:
        try (final GencodeFuncotationFactory funcotationFactory = new GencodeFuncotationFactory(new File(MUC16_GENCODE_TRANSCRIPT_FASTA_FILE))) {

            // Generate our funcotations:
            final List<GencodeFuncotation> funcotations = funcotationFactory.createFuncotations(variantContext, altAllele, gene, referenceContext);

            // Make sure we get what we expected:
            Assert.assertEquals(funcotations.size(), 1);

            final GencodeFuncotation funcotation = funcotations.get(0);

            Assert.assertEquals(funcotation.getVariantClassification(), expectedVariantClassification);
            Assert.assertEquals(funcotation.getVariantType(), expectedVariantType);
            Assert.assertEquals(funcotation.getGenomeChange(), expectedGenomeChange);
            Assert.assertEquals(funcotation.getTranscriptStrand(), expectedStrand);
            Assert.assertEquals(funcotation.getCodonChange(), expectedCodonChange);
            Assert.assertEquals(funcotation.getcDnaChange(), expectedCDnaChange);
            Assert.assertEquals(funcotation.getProteinChange(), expectedProteinChange);
            Assert.assertEquals(funcotation.getHugoSymbol(), expectedGeneName);
        }
    }

}
