package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import htsjdk.samtools.SAMSequenceRecord;
import htsjdk.samtools.reference.ReferenceSequence;
import htsjdk.tribble.Feature;
import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.ReferenceDataSource;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.funcotator.*;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * A factory to create {@link GencodeFuncotation}s.
 * Created by jonn on 8/30/17.
 */
public class GencodeFuncotationFactory extends DataSourceFuncotationFactory {

    /**
     * ReferenceSequenceFile for the transcript reference file.
     */
    private File fastaTranscriptFile;

    /**
     * ReferenceSequenceFile for the transcript reference file.
     */
    private ReferenceDataSource fastaTranscriptReferenceSequenceFile;

    /**
     * Map between transcript IDs and the IDs from the FASTA file to look up the transcript.
     * This is necessary because of the way the FASTA file contigs are named.
     */
    private Map<String, String> transcriptIdMap;

    //==================================================================================================================

    public GencodeFuncotationFactory(final File gencodeTranscriptFastaFile) {
        fastaTranscriptFile = gencodeTranscriptFastaFile;
        fastaTranscriptReferenceSequenceFile = ReferenceDataSource.of(gencodeTranscriptFastaFile);

        transcriptIdMap = createTranscriptIdMap(fastaTranscriptReferenceSequenceFile);
    }

    //==================================================================================================================

    @Override
    public void cleanup() {
        fastaTranscriptReferenceSequenceFile.close();
    }

    @Override
    public List<String> getSupportedFuncotationFields() {
        return GencodeFuncotation.getSerializedFieldNames();
    }

    @Override
    public List<Funcotation> createFuncotations(final VariantContext variant, final ReferenceContext referenceContext, final List<Feature> featureList) {
        final List<Funcotation> funcotations = new ArrayList<>();

        // If we have features we need to annotate, go through them and create annotations:
        if ( featureList.size() > 0 ) {
            for ( final Allele altAllele : variant.getAlternateAlleles() ) {
                for ( final Feature feature : featureList ) {

                    // Get the kind of feature we want here:
                    if ( GencodeGtfGeneFeature.class.isAssignableFrom(feature.getClass()) ) {
                        funcotations.addAll(createFuncotations(variant, altAllele, (GencodeGtfGeneFeature) feature, referenceContext));
                    }

                    // NOTE: If we don't have any funcotations for this feature, it's OK.
                    //       However, this means that some other DataSourceFuncotationFactory must be producing a
                    //       funcotation for this variant.
                    //       For it is decreed that all variants must have a funcotation, even if that funcotation be
                    //       empty.
                    // TODO: Actually you may want to put another IGR creation here for now...  This may be a more difficult thing if we determine it in here.  There is no way to know if these are IGRs or simply not included in this particular data set.
                }
            }
        }
        else {
            // This is an IGR.
            funcotations.addAll( createIgrFuncotations(variant, referenceContext) );
        }

        return funcotations;
    }

    //==================================================================================================================

    /**
     * Creates a map of Transcript IDs for use in looking up transcripts from the FASTA dictionary for the GENCODE Transcripts.
     * @param fastaReference The {@link ReferenceDataSource} corresponding to the Transcript FASTA file for this GENCODE dataset.
     * @return A {@link Map} of {@link String} -> {@link String} which maps real transcript IDs to the string of all transcript IDs for a given transcript sequence as it appears in the FASTA Transcript file.
     */
    private static Map<String, String> createTranscriptIdMap(final ReferenceDataSource fastaReference) {

        final Map<String, String> idMap = new HashMap<>();

        for ( final SAMSequenceRecord sequence : fastaReference.getSequenceDictionary().getSequences() ) {
            final String seqName = sequence.getSequenceName();

            // The names in the file are actually in a list with | between each sequence name.
            // We need to split the names and add them to the dictionary so we can resolve them to the full
            // sequence name as it appears in the file:
            for ( final String transcriptId : seqName.split("\\|") ) {
                idMap.put(transcriptId, seqName);
            }
        }

        return idMap;
    }

    /**
     * Creates a {@link List} of {@link GencodeFuncotation}s based on the given {@link VariantContext}, {@link Allele}, and {@link GencodeGtfGeneFeature}.
     * @param variant The variant to annotate.
     * @param altAllele The allele of the given variant to annotate.
     * @param gtfFeature The GTF feature on which to base annotations.
     * @return A {@link List} of {@link GencodeFuncotation}s for the given variant, allele and gtf feature.
     */
    private List<GencodeFuncotation> createFuncotations(final VariantContext variant, final Allele altAllele, final GencodeGtfGeneFeature gtfFeature, final ReferenceContext reference) {
        // For each applicable transcript, create an annotation.

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        final SimpleInterval variantPosition = new SimpleInterval(variant.getContig(), variant.getStart(), variant.getEnd());

        // Get our "best" transcript:
        final int bestTranscriptIndex = getBestTranscriptIndex(gtfFeature, variant);
        if ( bestTranscriptIndex == -1 ) {
            throw new GATKException("Could not get a good transcript for the given feature: " + gtfFeature.toString());
        }
        final GencodeGtfTranscriptFeature transcript = gtfFeature.getTranscripts().remove(bestTranscriptIndex);

        // Setup the "trivial" fields of the gencodeFuncotation:
        final GencodeFuncotation gencodeFuncotation = createGencodeFuncotationWithTrivialFieldsPopulated(variant, altAllele, gtfFeature, transcript);

        // Get the list of exons by their locations so we can use them to determine our location in the transcript and get
        // the transcript code itself:
        final List<? extends htsjdk.samtools.util.Locatable> exonPositionList =
                transcript.getExons().stream()
                        .filter(e -> (e.getCds() != null))
                        .map(GencodeGtfExonFeature::getCds)
                        .collect(Collectors.toList());

        // Set up our SequenceComparison object so we can calculate some useful fields more easily
        // These fields can all be set without knowing the alternate allele:
        final FuncotatorUtils.SequenceComparison sequenceComparison = createSequenceComparisonWithTrivialFieldsPopulated(variant, reference, transcript, exonPositionList);




//        boolean hasBeenAnnotated = false;
//
//        // Find the exon in which we have the variant:
//        for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
//            if ( exon.getGenomicPosition().overlaps(variantPosition) ) {
//
//                // Set our transcript exon number:
//                gencodeFuncotation.setTranscriptExon( exon.getExonNumber() );
//
//                sequenceComparison.setAlignedAlternateAlleleStop(
//                        FuncotatorUtils.getAlignedEndPosition(
//                                sequenceComparison.getAlignedCodingSequenceAlleleStart(),
//                                altAllele.length()
//                        )
//                );
//
//                final String altCodingSequence = FuncotatorUtils.getAlternateCodingSequence(
//                        sequenceComparison.getWholeReferenceSequence().getBaseString(),
//                        sequenceComparison.getCodingSequenceAlleleStart(),
//                        variant.getReference(),
//                        altAllele );
//
//                final String altAminoAcidSequence = FuncotatorUtils.createAminoAcidSequence(altCodingSequence);
//
//                // Note we add 1 because substring ends are EXCLUSIVE:
//                sequenceComparison.setAlignedAlternateAllele(
//                        altCodingSequence.substring(
//                                sequenceComparison.getAlignedCodingSequenceAlleleStart(),
//                                sequenceComparison.getAlignedAlternateAlleleStop() + 1
//                        )
//                );
//
//                sequenceComparison.setAlternateAminoAcidSequence(
//                        FuncotatorUtils.createAminoAcidSequence( sequenceComparison.getAlignedAlternateAllele() )
//                );
//
//                sequenceComparison.setProteinChangeEndPosition(
//                        sequenceComparison.getProteinChangeStartPosition() + (sequenceComparison.getAlignedAlternateAllele().length() / 3)
//                );
//
//                gencodeFuncotation.setCodonChange( FuncotatorUtils.getCodonChangeString(sequenceComparison) );
//                gencodeFuncotation.setProteinChange( FuncotatorUtils.getProteinChangeString(sequenceComparison) );
//                gencodeFuncotation.setcDnaChange( FuncotatorUtils.getCodingSequenceChangeString(sequenceComparison) );
//
////        TODO: FIVE_PRIME_FLANK(15),
////        TODO: THREE_PRIME_FLANK(15),
//
////        TODO: RNA(4),
////        TODO: LINCRNA(4);
//
//                // Determine the variant classification:
//                if ( FuncotatorUtils.isFrameshift(variant.getReference(), altAllele)) {
//
//                    if ( variant.getReference().length() < altAllele.length() ) {
//                        gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS);
//                    }
//                    else {
//                        gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL);
//                    }
//
//                    // check for de novo starts out of frame:
//                    for ( int i = 0 ; i < sequenceComparison.getAlternateAminoAcidSequence().length(); ++i) {
//                        final char aa = sequenceComparison.getAlternateAminoAcidSequence().charAt(i);
//                        if ( aa == AminoAcid.METHIONINE.getLetter().charAt(0) ) {
//                            gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.DE_NOVO_START_OUT_FRAME);
//                            break;
//                        }
//                    }
//                }
//                else {
//
//                    // Check for in frame indels:
//                    if ( variant.getReference().length() < altAllele.length() ) {
//                        gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.IN_FRAME_INS);
//                    }
//                    else if ( variant.getReference().length() < altAllele.length() ) {
//                        gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.IN_FRAME_DEL);
//                    }
//                    else {
//                        // We know the sequences are equal.
//                        if ( sequenceComparison.getReferenceAminoAcidSequence().equals(sequenceComparison.getAlternateAminoAcidSequence())) {
//                            gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.SILENT);
//                            if (FuncotatorUtils.isSpliceSiteVariant(variant, exonPositionList)) {
//                                gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.SPLICE_SITE);
//                            }
//                        }
//                        else {
//                            //Now we check the sequence itself for problems:
//                            if ( (exon.getStartCodon() != null) && (new SimpleInterval(exon.getStartCodon()).overlaps(
//                                    new SimpleInterval(variant.getContig(), variant.getStart(), variant.getStart() + altAllele.length() - 1))) ) {
//                                // Start codon mutation!
//                                if ( variant.getReference().length() < altAllele.length() ) {
//                                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.START_CODON_INS);
//                                }
//                                else if ( variant.getReference().length() > altAllele.length() ) {
//                                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.START_CODON_DEL);
//                                }
//                                else {
//                                    // TODO: Strictly speaking, this is not correct - we need to make sure that we have only a single codon changed:
//                                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.START_CODON_SNP);
//                                }
//                            }
//
//                            // check for de novo start in frame and missense:
//                            for ( int i = 0 ; i < sequenceComparison.getAlternateAminoAcidSequence().length(); ++i) {
//                                final char altAa = sequenceComparison.getAlternateAminoAcidSequence().charAt(i);
//                                final char refAa = sequenceComparison.getReferenceAminoAcidSequence().charAt(i);
//                                if ( altAa == AminoAcid.METHIONINE.getLetter().charAt(0) ) {
//                                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.DE_NOVO_START_IN_FRAME);
//                                }
//                                if ( altAa != refAa ) {
//                                    // Missense trumps De novo start in frame, so if we get one we can get out:
//                                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.MISSENSE);
//                                    break;
//                                }
//                            }
//
//                            // Check for nonsense:
//                            if ( altAminoAcidSequence.contains(AminoAcid.NONSENSE.getLetter()) ) {
//                                gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.NONSENSE);
//                            }
//                        }
//                    }
//                }
//
//                // check for non-stop here since it's the worst of the worst:
//                if ( FuncotatorUtils.isNonStopMutant(altAminoAcidSequence) ) {
//                    gencodeFuncotation.setVariantClassification(GencodeFuncotation.VariantClassification.NONSTOP);
//                }
//
//                hasBeenAnnotated = true;
//                break;
//            }
//        }
//
//        if ( !hasBeenAnnotated ) {
//            // We know that this must be an intron or in the UTR because we know we are within the bounds of a GencodeGtfGeneFeature
//            // and that we did not fall inside any of the exons:
//
//            // Check for UTRs:
//            for ( final GencodeGtfUTRFeature utr : transcript.getUtrs() ) {
//                if ( new SimpleInterval(utr).overlaps(variant) ) {
//                    if ( is3PrimeUtr(utr, transcript) ) {
//                        gencodeFuncotation.setVariantClassification( GencodeFuncotation.VariantClassification.THREE_PRIME_UTR );
//                    }
//                    else {
//                        gencodeFuncotation.setVariantClassification( GencodeFuncotation.VariantClassification.FIVE_PRIME_UTR );
//                    }
//                    hasBeenAnnotated = true;
//                }
//            }
//
//            // Check for introns:
//            if ( !hasBeenAnnotated ) {
//                gencodeFuncotation.setVariantClassification( GencodeFuncotation.VariantClassification.INTRON );
//            }
//        }

        gencodeFuncotations.add(gencodeFuncotation);
        return gencodeFuncotations;
    }

    /**
     * Determine what type of region the given variant lies in given a transcript.
     * @param variant A {@link VariantContext} of which to determine the region.
     * @param transcript A {@link GencodeGtfTranscriptFeature} to use to determine the region of {@code variant}.
     * @return The {@link FuncotatorUtils.VariantGenomeRegionType} corresponding to the region of {@code variant} given {@code transcript}.
     */
    private FuncotatorUtils.VariantGenomeRegionType determineGenomeRegionType(final VariantContext variant, final GencodeGtfTranscriptFeature transcript) {

        if ( new SimpleInterval(transcript).contains(variant) ) {
            // Check if we have the mutation in a UTR first:
            if ( transcript.getUtrs().size() > 0 ) {
                for ( final GencodeGtfUTRFeature utr : transcript.getUtrs() ) {
                    if ( new SimpleInterval(utr).contains(variant) ) {

                    }
                }
            }
        }
        else {
            return FuncotatorUtils.VariantGenomeRegionType.IGR;
        }

    }

    /**
     * Creates a {@link org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils.SequenceComparison} object with the trivial fields populated.
     * @param variant The {@link VariantContext} for the current variant.
     * @param reference The {@link ReferenceContext} for the current sample set.
     * @param transcript The {@link GencodeGtfTranscriptFeature} for the current gene feature / alt allele.
     * @param exonPositionList A {@link List} of {@link htsjdk.samtools.util.Locatable} objects representing exon positions in the transcript.
     * @return A trivially populated {@link org.broadinstitute.hellbender.tools.funcotator.FuncotatorUtils.SequenceComparison} object.
     */
    private FuncotatorUtils.SequenceComparison createSequenceComparisonWithTrivialFieldsPopulated(final VariantContext variant,
                                                                                                  final ReferenceContext reference,
                                                                                                  final GencodeGtfTranscriptFeature transcript,
                                                                                                  final List<? extends htsjdk.samtools.util.Locatable> exonPositionList) {

        final FuncotatorUtils.SequenceComparison sequenceComparison = new FuncotatorUtils.SequenceComparison();

        // Get the contig:
        sequenceComparison.setContig(variant.getContig());

        // Get the reference sequence in the coding region as described by the given exonPositionList:
        sequenceComparison.setWholeReferenceSequence(new ReferenceSequence(transcript.getTranscriptId(),transcript.getStart(),FuncotatorUtils.getCodingSequence(reference, exonPositionList).getBytes()));

        // Get the ref allele:
        sequenceComparison.setReferenceAllele(variant.getReference().getBaseString());

        // Get the allele genomic start position:
        sequenceComparison.setAlleleStart(variant.getStart());

        // Get the allele transcript start position:
        sequenceComparison.setTranscriptAlleleStart(variant.getStart() - transcript.getStart());

        // Get the coding region start position (in the above computed reference coding region):
        sequenceComparison.setCodingSequenceAlleleStart(FuncotatorUtils.getStartPositionInTranscript(variant, exonPositionList));

        // Get the in-frame start position of the codon containing the given variant:
        sequenceComparison.setAlignedCodingSequenceAlleleStart(FuncotatorUtils.getAlignedPosition(sequenceComparison.getCodingSequenceAlleleStart()));

        // Get the in-frame stop position of the codon containing the given variant:
        sequenceComparison.setAlignedReferenceAlleleStop(FuncotatorUtils.getAlignedEndPosition(sequenceComparison.getAlignedCodingSequenceAlleleStart(),
                variant.getReference().length() ));

        // Get the in-frame/codon-aligned region containing the reference allele:
        sequenceComparison.setAlignedReferenceAllele(
                sequenceComparison.getWholeReferenceSequence().getBaseString().substring(
                        sequenceComparison.getAlignedCodingSequenceAlleleStart(),
                        sequenceComparison.getAlignedReferenceAlleleStop() + 1 // Add 1 because the stop position is inclusive
                )
        );

        // Get the amino acid sequence of the reference allele:
        sequenceComparison.setReferenceAminoAcidSequence(
                FuncotatorUtils.createAminoAcidSequence( sequenceComparison.getAlignedReferenceAllele() )
        );

        // Get the starting protein position of this variant.
        sequenceComparison.setProteinChangeStartPosition(
                sequenceComparison.getAlignedCodingSequenceAlleleStart() / 3
        );

        return sequenceComparison;
    }

    /**
     * Creates a Gencode Funcotation with all trivial fields populated.
     * @param variant The {@link VariantContext} for the current variant.
     * @param altAllele The alternate {@link Allele} we are currently annotating.
     * @param gtfFeature The current {@link GencodeGtfGeneFeature} read from the input feature file.
     * @param transcript The current {@link GencodeGtfTranscriptFeature} containing our {@code altAllele}.
     * @return A trivially populated {@link GencodeFuncotation} object.
     */
    private GencodeFuncotation createGencodeFuncotationWithTrivialFieldsPopulated(final VariantContext variant,
                                                                                  final Allele altAllele,
                                                                                  final GencodeGtfGeneFeature gtfFeature,
                                                                                  final GencodeGtfTranscriptFeature transcript) {
        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        gencodeFuncotation.setHugoSymbol( gtfFeature.getGeneName() );
        gencodeFuncotation.setChromosome( gtfFeature.getChromosomeName() );

        gencodeFuncotation.setOtherTranscripts(
                gtfFeature.getTranscripts().stream().map(GencodeGtfTranscriptFeature::getTranscriptId).collect(Collectors.toList())
        );

        gencodeFuncotation.setStart(variant.getStart());

        // The end position is inclusive, so we need to make sure we don't double-count the start position (so we subtract 1):
        gencodeFuncotation.setEnd(variant.getStart() + altAllele.length() - 1);

        gencodeFuncotation.setVariantType( getVariantType(variant.getReference(), altAllele) );
        gencodeFuncotation.setRefAllele( variant.getReference().getBaseString() );
        gencodeFuncotation.setTumorSeqAllele1( altAllele.getBaseString() );
        gencodeFuncotation.setTumorSeqAllele2( altAllele.getBaseString() );
        gencodeFuncotation.setAnnotationTranscript( transcript.getTranscriptId() );
        gencodeFuncotation.setTranscriptStrand( transcript.getGenomicStrand().toString() );
        gencodeFuncotation.setGenomeChange(getGenomeChangeString(variant, altAllele, gtfFeature));
        gencodeFuncotation.setTranscriptPos( variant.getStart() - transcript.getStart() );

        gencodeFuncotation.setNcbiBuild( gtfFeature.getUcscGenomeVersion() );

        return gencodeFuncotation;
    }

    /**
     * Determines if the given UTR is 3' or 5' of the given transcript.
     * Assumes the UTR is part of the given transcript.
     * @param utr The {@link GencodeGtfUTRFeature} to check for relative location in the given {@link GencodeGtfTranscriptFeature}.
     * @param transcript The {@link GencodeGtfTranscriptFeature} in which to check for the given {@code utr}.
     * @return {@code true} if the given {@code utr} is 3' for the given {@code transcript}; {@code false} otherwise.
     */
    private boolean is3PrimeUtr(final GencodeGtfUTRFeature utr, final GencodeGtfTranscriptFeature transcript) {
        boolean isBefore = true;
        if ( transcript.getGenomicStrand() == GencodeGtfFeature.GenomicStrand.FORWARD ) {
            for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
                if ( exon.getStart() < utr.getStart()) {
                    isBefore = false;
                    break;
                }
            }
        }
        else {
            for ( final GencodeGtfExonFeature exon : transcript.getExons() ) {
                if ( exon.getStart() > utr.getStart()) {
                    isBefore = false;
                    break;
                }
            }
        }

        return isBefore;
    }

    /**
     * Creates a string representing the genome change given the variant, allele, and gene feature for this variant.
     * @param variant {@link VariantContext} of which to create the change.
     * @param altAllele {@link Allele} representing the alternate allele for this variant.
     * @param gtfFeature {@link GencodeGtfGeneFeature} corresponding to this variant.
     * @return A short {@link String} representation of the genomic change for the given variant, allele, and feature.
     */
    private String getGenomeChangeString(final VariantContext variant, final Allele altAllele, final GencodeGtfGeneFeature gtfFeature) {
        return "g." + gtfFeature.getChromosomeName() +
                ":" + variant.getStart() +
                variant.getReference().getBaseString() + ">" + altAllele.getBaseString();
    }

    /**
     * Return the index of the "best" transcript in this gene.
     * @param geneFeature A {@link GencodeGtfGeneFeature} from which to get the index of the "best" transcript.
     * @param variant The {@link VariantContext} for which we want to get the best index.
     * @return The index of the "best" {@link GencodeGtfTranscriptFeature} in the given {@link GencodeGtfGeneFeature}.  Returns -1 if no transcript is present.
     */
    private int getBestTranscriptIndex(final GencodeGtfGeneFeature geneFeature, final VariantContext variant) {
        if ( geneFeature.getTranscripts().size() == 0 ) {
            return -1;
        }

        for ( int i = 0 ; i < geneFeature.getTranscripts().size() ; ++i ) {
            if ( geneFeature.getTranscripts().get(i).getGenomicPosition().overlaps(variant) ) {
                return i;
            }
        }

        // Oops... we didn't find anything.
        return -1;
    }

    /**
     * Creates a {@link List} of {@link GencodeFuncotation}s based on the given {@link VariantContext}.
     * These are most likely to be {@link GencodeFuncotation.VariantClassification#IGR}, but could also be
     * {@link GencodeFuncotation.VariantClassification#DE_NOVO_START_OUT_FRAME}
     * @param variant The variant to annotate.
     * @param reference The reference against which to compare the given variant.
     * @return A list of IGR annotations for the given variant.
     */
    private List<GencodeFuncotation> createIgrFuncotations(final VariantContext variant, final ReferenceContext reference) {
        // for each allele, create an annotation.

        final List<GencodeFuncotation> gencodeFuncotations = new ArrayList<>();

        final int windowSizeInBases = 3;

        reference.setWindow(windowSizeInBases,windowSizeInBases);
        final String referenceSequence = new String( reference.getBases() );

        for ( final Allele allele : variant.getAlternateAlleles() ) {
            gencodeFuncotations.add(
                    createIgrAnnotationForAllele(allele,
                            referenceSequence.substring(0,windowSizeInBases) +
                            allele.getBaseString() +
                            referenceSequence.substring(windowSizeInBases + variant.getReference().length()))
            );
        }

        return gencodeFuncotations;
    }

    /**
     * Create a {@link GencodeFuncotation} representing the intergenic region variant given by {@code allele} and the {@code reference}.
     * @param altAllele An alternate {@link Allele} from which to create a {@link GencodeFuncotation}.
     * @param newSequence A {@link String} representing the new genetic sequence with the given alternate allele.
     * @return A {@link GencodeFuncotation} corresponding to an IGR variant given an allele and reference.
     */
    private GencodeFuncotation createIgrAnnotationForAllele(final Allele altAllele, final String newSequence) {

        final GencodeFuncotation gencodeFuncotation = new GencodeFuncotation();

        // Determine if we're an IGR or a DE_NOVO_START:
        GencodeFuncotation.VariantClassification classification = GencodeFuncotation.VariantClassification.IGR;

        if ( newSequence.contains(AminoAcid.METHIONINE.getCodons()[0])) {
            classification = GencodeFuncotation.VariantClassification.DE_NOVO_START_OUT_FRAME;
        }

        gencodeFuncotation.setVariantClassification( classification );
        gencodeFuncotation.setTumorSeqAllele1( altAllele.getBaseString() );
        gencodeFuncotation.setTumorSeqAllele2( altAllele.getBaseString() );

        return gencodeFuncotation;
    }

    /**
     * Determines the variant type based on the given reference allele and alternate allele.
     * @param refAllele The reference {@link Allele} for this variant.
     * @param altAllele The alternate {@link Allele} for this variant.
     * @return A {@link GencodeFuncotation.VariantType} representing the variation type between the given reference and alternate {@link Allele}.
     */
    private GencodeFuncotation.VariantType getVariantType( final Allele refAllele, final Allele altAllele ) {

        if ( altAllele.length() > refAllele.length() ) {
            return GencodeFuncotation.VariantType.INS;
        }
        else if (altAllele.length() < refAllele.length()) {
            return GencodeFuncotation.VariantType.DEL;
        }
        else {
            // We know they are the same length, now we just need to check one of them:
            switch (refAllele.length()) {
                case 1:  return GencodeFuncotation.VariantType.SNP;
                case 2:  return GencodeFuncotation.VariantType.DNP;
                case 3:  return GencodeFuncotation.VariantType.TNP;
                default: return GencodeFuncotation.VariantType.ONP;
            }
        }
    }

     /**
     * Calculates the offset required to get the aligned codon start for the given variant.
     * @param variant {@link VariantContext} to align.
     * @param transcript {@link GencodeGtfTranscriptFeature} against which to align the given {@link VariantContext}
     * @return An offset which when subtracted from the variant start position will give the start of the codon containing the first base of the variant.
     */
    private int calculateOffsetToTranscriptAlignment(final VariantContext variant, final GencodeGtfTranscriptFeature transcript) {
        if ( !transcript.getGenomicPosition().overlaps(variant) ) {
            return 0;
        }

        return ((variant.getStart() - transcript.getStart()) % 3);
    }
}
