package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import org.broadinstitute.hellbender.utils.codecs.GENCODE.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * A class to hold a method that provides a valid {@link GencodeGtfGeneFeature} object for testing.
 * Taken and modified from the unit tests for the GENCODE GTF Tribble Codec.
 * Created by jonn on 10/3/17.
 */
public class DataProviderForExampleGencodeGtfGene {

    public static GencodeGtfGeneFeature createGencodeGtfGeneFeature() {

        // Let's define all our features up front:
        GencodeGtfFeatureBaseData data;

        data = new GencodeGtfFeatureBaseData(6, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.GENE,
                50200979, 50217616, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", null, GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", null, null, null, -1, null, GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Collections.singletonList(
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfGeneFeature gene1 = (GencodeGtfGeneFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(7, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.TRANSCRIPT,
                50200979, 50217615, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", -1, null, GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfTranscriptFeature transcript1 = (GencodeGtfTranscriptFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(8, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50200979, 50201590, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon1 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(9, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50201037, 50201590, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds1 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(10, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.START_CODON,
                50201037, 50201039, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfStartCodonFeature start_codon1 = (GencodeGtfStartCodonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(11, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50206317, 50206520, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 2, "ENSE00001129529.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon2 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(12, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50206317, 50206520, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 2, "ENSE00001129529.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds2 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(13, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50208536, 50208716, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 3, "ENSE00001129524.2", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon3 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(14, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50208536, 50208716, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 3, "ENSE00001129524.2", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds3 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(15, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50210181, 50210311, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 4, "ENSE00003473644.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon4 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(16, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50210181, 50210311, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 4, "ENSE00003473644.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds4 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(17, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50210631, 50210911, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 5, "ENSE00003503715.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon5 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(18, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50210631, 50210911, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 5, "ENSE00003503715.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds5 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(19, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50215717, 50215867, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 6, "ENSE00003573348.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon6 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(20, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50215717, 50215867, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.TWO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 6, "ENSE00003573348.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds6 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(21, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50216691, 50216876, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 7, "ENSE00003510005.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon7 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(22, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50216691, 50216876, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 7, "ENSE00003510005.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds7 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(23, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50216972, 50217128, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 8, "ENSE00003591346.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon8 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(24, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50216972, 50217128, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 8, "ENSE00003591346.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds8 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(25, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50217205, 50217357, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 9, "ENSE00003728455.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon9 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(26, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50217205, 50217357, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 9, "ENSE00003728455.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds9 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(27, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.EXON,
                50217361, 50217615, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 10, "ENSE00003739808.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon10 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(28, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.CDS,
                50217361, 50217366, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 10, "ENSE00003739808.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds10 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(29, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.STOP_CODON,
                50217367, 50217369, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 10, "ENSE00003739808.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfStopCodonFeature stop_codon1 = (GencodeGtfStopCodonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(30, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.UTR,
                50200979, 50201036, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfUTRFeature utr1 = (GencodeGtfUTRFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(31, "chr22", GencodeGtfFeature.AnnotationSource.ENSEMBL, GencodeGtfFeature.FeatureType.UTR,
                50217367, 50217615, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000611222.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-201", 10, "ENSE00003739808.1", GencodeGtfFeature.LocusLevel.AUTOMATICALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000483593.1"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.NO_SINGLE_TRANSCRIPT_SUPPORT),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_ALTERNATIVE_2),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3")
                        )
                ),
                null
        );
        final GencodeGtfUTRFeature utr2 = (GencodeGtfUTRFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(32, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.TRANSCRIPT,
                50200979, 50217616, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", -1, null, GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfTranscriptFeature transcript2 = (GencodeGtfTranscriptFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(33, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.SELENOCYSTEINE,
                50217358, 50217360, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", -1, null, GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfSelenocysteineFeature selenocysteine1 = (GencodeGtfSelenocysteineFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(34, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50200979, 50201590, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon11 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(35, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50201037, 50201590, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds11 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(36, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.START_CODON,
                50201037, 50201039, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfStartCodonFeature start_codon2 = (GencodeGtfStartCodonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(37, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50206317, 50206520, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 2, "ENSE00001129529.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon12 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(38, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50206317, 50206520, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 2, "ENSE00001129529.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds12 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(39, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50208536, 50208716, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 3, "ENSE00001129524.2", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon13 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(40, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50208536, 50208716, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 3, "ENSE00001129524.2", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds13 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(41, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50210181, 50210311, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 4, "ENSE00003473644.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon14 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(42, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50210181, 50210311, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 4, "ENSE00003473644.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds14 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(43, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50210631, 50210911, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 5, "ENSE00003503715.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon15 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(44, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50210631, 50210911, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 5, "ENSE00003503715.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds15 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(45, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50215717, 50215867, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 6, "ENSE00003573348.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon16 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(46, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50215717, 50215867, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.TWO, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 6, "ENSE00003573348.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds16 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(47, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50216691, 50216876, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 7, "ENSE00003510005.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon17 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(48, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50216691, 50216876, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 7, "ENSE00003510005.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds17 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(49, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50216972, 50217128, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 8, "ENSE00003591346.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon18 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(50, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50216972, 50217128, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ONE, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 8, "ENSE00003591346.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds18 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(51, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50217205, 50217616, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 9, "ENSE00003512975.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon19 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(52, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.CDS,
                50217205, 50217366, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 9, "ENSE00003512975.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfCDSFeature cds19 = (GencodeGtfCDSFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(53, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.STOP_CODON,
                50217367, 50217369, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.ZERO, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 9, "ENSE00003512975.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfStopCodonFeature stop_codon2 = (GencodeGtfStopCodonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(54, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.UTR,
                50200979, 50201036, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 1, "ENSE00001541223.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfUTRFeature utr3 = (GencodeGtfUTRFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(55, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.UTR,
                50217367, 50217616, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000380903.6", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING, null, "SELENOO-001", 9, "ENSE00003512975.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("protein_id", "ENSP00000370288.2"),
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.BASIC),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.APPRIS_PRINCIPAL_2),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.CCDS),
                                new GencodeGtfFeature.OptionalField<>("tag", GencodeGtfFeature.FeatureTag.SELENO),
                                new GencodeGtfFeature.OptionalField<>("ccdsid", "CCDS43034.1"),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000075003.2")
                        )
                ),
                null
        );
        final GencodeGtfUTRFeature utr4 = (GencodeGtfUTRFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(56, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.TRANSCRIPT,
                50206442, 50217616, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", -1, null, GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfTranscriptFeature transcript3 = (GencodeGtfTranscriptFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(57, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50206442, 50206520, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 1, "ENSE00001890724.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon20 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(58, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50208488, 50208716, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 2, "ENSE00001952603.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon21 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(59, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50210181, 50210311, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 3, "ENSE00003583919.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon22 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(60, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50210631, 50210911, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 4, "ENSE00003620115.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon23 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(61, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50215717, 50215867, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 5, "ENSE00003636069.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon24 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(62, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50216691, 50216876, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 6, "ENSE00003579717.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon25 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(63, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50216972, 50217128, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 7, "ENSE00003650938.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon26 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);
        data = new GencodeGtfFeatureBaseData(64, "chr22", GencodeGtfFeature.AnnotationSource.HAVANA, GencodeGtfFeature.FeatureType.EXON,
                50217205, 50217616, GencodeGtfFeature.GenomicStrand.FORWARD, GencodeGtfFeature.GenomicPhase.DOT, "ENSG00000073169.13", "ENST00000492092.1", GencodeGtfFeature.GeneTranscriptType.PROTEIN_CODING,
                null, "SELENOO", GencodeGtfFeature.GeneTranscriptType.PROCESSED_TRANSCRIPT, null, "SELENOO-002", 8, "ENSE00003475904.1", GencodeGtfFeature.LocusLevel.MANUALLY_ANNOTATED,
                new ArrayList<>(
                        Arrays.asList(
                                new GencodeGtfFeature.OptionalField<>("transcript_support_level", GencodeGtfFeature.TranscriptSupportLevel.ALL_MRNA_VERIFIED),
                                new GencodeGtfFeature.OptionalField<>("havana_gene", "OTTHUMG00000044645.3"),
                                new GencodeGtfFeature.OptionalField<>("havana_transcript", "OTTHUMT00000316993.1")
                        )
                ),
                null
        );
        final GencodeGtfExonFeature exon27 = (GencodeGtfExonFeature) GencodeGtfFeature.create(data);

        // ======================
        // Here we'll set the version number of each feature.
        // Normally this is set in the decode method, so we need to do it manually.

        cds1.setUcscGenomeVersion("hg38");
        cds2.setUcscGenomeVersion("hg38");
        cds3.setUcscGenomeVersion("hg38");
        cds4.setUcscGenomeVersion("hg38");
        cds5.setUcscGenomeVersion("hg38");
        cds6.setUcscGenomeVersion("hg38");
        cds7.setUcscGenomeVersion("hg38");
        cds8.setUcscGenomeVersion("hg38");
        cds9.setUcscGenomeVersion("hg38");
        cds10.setUcscGenomeVersion("hg38");
        cds11.setUcscGenomeVersion("hg38");
        cds12.setUcscGenomeVersion("hg38");
        cds13.setUcscGenomeVersion("hg38");
        cds14.setUcscGenomeVersion("hg38");
        cds15.setUcscGenomeVersion("hg38");
        cds16.setUcscGenomeVersion("hg38");
        cds17.setUcscGenomeVersion("hg38");
        cds18.setUcscGenomeVersion("hg38");
        cds19.setUcscGenomeVersion("hg38");
        exon1.setUcscGenomeVersion("hg38");
        exon2.setUcscGenomeVersion("hg38");
        exon3.setUcscGenomeVersion("hg38");
        exon4.setUcscGenomeVersion("hg38");
        exon5.setUcscGenomeVersion("hg38");
        exon6.setUcscGenomeVersion("hg38");
        exon7.setUcscGenomeVersion("hg38");
        exon8.setUcscGenomeVersion("hg38");
        exon9.setUcscGenomeVersion("hg38");
        exon10.setUcscGenomeVersion("hg38");
        exon11.setUcscGenomeVersion("hg38");
        exon12.setUcscGenomeVersion("hg38");
        exon13.setUcscGenomeVersion("hg38");
        exon14.setUcscGenomeVersion("hg38");
        exon15.setUcscGenomeVersion("hg38");
        exon16.setUcscGenomeVersion("hg38");
        exon17.setUcscGenomeVersion("hg38");
        exon18.setUcscGenomeVersion("hg38");
        exon19.setUcscGenomeVersion("hg38");
        exon20.setUcscGenomeVersion("hg38");
        exon21.setUcscGenomeVersion("hg38");
        exon22.setUcscGenomeVersion("hg38");
        exon23.setUcscGenomeVersion("hg38");
        exon24.setUcscGenomeVersion("hg38");
        exon25.setUcscGenomeVersion("hg38");
        exon26.setUcscGenomeVersion("hg38");
        exon27.setUcscGenomeVersion("hg38");
        start_codon1.setUcscGenomeVersion("hg38");
        stop_codon1.setUcscGenomeVersion("hg38");
        start_codon2.setUcscGenomeVersion("hg38");
        stop_codon2.setUcscGenomeVersion("hg38");
        transcript1.setUcscGenomeVersion("hg38");
        transcript2.setUcscGenomeVersion("hg38");
        transcript3.setUcscGenomeVersion("hg38");
        utr1.setUcscGenomeVersion("hg38");
        utr2.setUcscGenomeVersion("hg38");
        utr3.setUcscGenomeVersion("hg38");
        utr4.setUcscGenomeVersion("hg38");
        selenocysteine1.setUcscGenomeVersion("hg38");
        gene1.setUcscGenomeVersion("hg38");

        // ======================
        // Now let's collapse these objects into their correct structure:

        exon1.setCds(cds1);
        exon1.setStartCodon(start_codon1);

        exon2.setCds(cds2);
        exon3.setCds(cds3);
        exon4.setCds(cds4);
        exon5.setCds(cds5);
        exon6.setCds(cds6);
        exon7.setCds(cds7);
        exon8.setCds(cds8);
        exon9.setCds(cds9);

        exon10.setCds(cds10);
        exon10.setStopCodon(stop_codon1);

        transcript1.addExon(exon1);
        transcript1.addExon(exon2);
        transcript1.addExon(exon3);
        transcript1.addExon(exon4);
        transcript1.addExon(exon5);
        transcript1.addExon(exon6);
        transcript1.addExon(exon7);
        transcript1.addExon(exon8);
        transcript1.addExon(exon9);
        transcript1.addExon(exon10);

        transcript1.addUtr(utr1);
        transcript1.addUtr(utr2);

        gene1.addTranscript(transcript1);

        // ======================

        transcript2.addSelenocysteine(selenocysteine1);

        exon11.setStartCodon(start_codon2);

        exon12.setCds(cds12);
        exon13.setCds(cds13);
        exon14.setCds(cds14);
        exon15.setCds(cds15);
        exon16.setCds(cds16);
        exon17.setCds(cds17);
        exon18.setCds(cds18);

        exon19.setStopCodon(stop_codon2);

        transcript2.addExon(exon11);
        transcript2.addExon(exon12);
        transcript2.addExon(exon13);
        transcript2.addExon(exon14);
        transcript2.addExon(exon15);
        transcript2.addExon(exon16);
        transcript2.addExon(exon17);
        transcript2.addExon(exon18);
        transcript2.addExon(exon19);

        transcript2.addUtr(utr3);
        transcript2.addUtr(utr4);

        gene1.addTranscript(transcript2);

        // ======================

        transcript3.addExon(exon20);
        transcript3.addExon(exon21);
        transcript3.addExon(exon22);
        transcript3.addExon(exon23);
        transcript3.addExon(exon24);
        transcript3.addExon(exon25);
        transcript3.addExon(exon26);
        transcript3.addExon(exon27);

        gene1.addTranscript(transcript3);

        // ======================

        return gene1;
    }


}
