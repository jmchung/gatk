package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import com.google.common.annotations.VisibleForTesting;
import org.broadinstitute.hellbender.tools.funcotator.Funcotation;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;

/**
 * A class to represent a Functional Annotation.
 * Created by jonn on 8/22/17.
 */
public class GencodeFuncotation extends Funcotation {

    @VisibleForTesting
    static final String FIELD_DELIMITER = "|";

    //==================================================================================================================

    private String                  hugoSymbol;
    private String                  ncbiBuild;
    private String                  chromosome;
    private int                     start;
    private int                     end;
    private VariantClassification   variantClassification;
    private VariantType             variantType;
    private String                  refAllele;
    private String                  tumorSeqAllele1;
    private String                  tumorSeqAllele2;

    private String                  genomeChange;
    private String                  annotationTranscript;
    private String                  transcriptStrand;
    private int                     transcriptExon;
    private int                     transcriptPos;
    private String                  cDnaChange;
    private String                  codonChange;
    private String                  proteinChange;
    private List<String>            otherTranscripts;

    //==================================================================================================================

    /**
     * Basic constructor for a {@link GencodeFuncotation}.
     */
    public GencodeFuncotation() {}

    //==================================================================================================================

    /**
     * @return An ordered {@link List} of {@link String} containing the field names that {@link GencodeFuncotation} produces.
     */
    public static List<String> getSerializedFieldNames() {

        final List<String> fields = new ArrayList<>();

        for(final Field f : GencodeFuncotation.class.getDeclaredFields() ) {
            if ( !Modifier.isStatic(f.getModifiers()) ) {
                fields.add( f.getName() );
            }
        }

        return fields;
    }

    //==================================================================================================================

    /**
     * Converts this {@link GencodeFuncotation} to a string suitable for insertion into a VCF file.
     * @return a {@link String} representing this {@link GencodeFuncotation} suitable for insertion into a VCF file.
     */
    @Override
    public String serializeToVcfString() {

        return (hugoSymbol != null ? hugoSymbol : "") + FIELD_DELIMITER +
                (ncbiBuild != null ? ncbiBuild : "") + FIELD_DELIMITER +
                (chromosome != null ? chromosome : "") + FIELD_DELIMITER +
                start + FIELD_DELIMITER +
                end + FIELD_DELIMITER +
                (variantClassification != null ? variantClassification : "") + FIELD_DELIMITER +
                (variantType != null ? variantType : "") + FIELD_DELIMITER +
                (refAllele != null ? refAllele : "") + FIELD_DELIMITER +
                (tumorSeqAllele1 != null ? tumorSeqAllele1 : "") + FIELD_DELIMITER +
                (tumorSeqAllele2 != null ? tumorSeqAllele2 : "") + FIELD_DELIMITER +
                (genomeChange != null ? genomeChange : "") + FIELD_DELIMITER +
                (annotationTranscript != null ? annotationTranscript : "") + FIELD_DELIMITER +
                (transcriptStrand != null ? transcriptStrand : "") + FIELD_DELIMITER +
                transcriptExon + FIELD_DELIMITER +
                transcriptPos + FIELD_DELIMITER +
                (cDnaChange != null ? cDnaChange : "") + FIELD_DELIMITER +
                (codonChange != null ? codonChange : "") + FIELD_DELIMITER +
                (proteinChange != null ? proteinChange : "") + FIELD_DELIMITER +
                (otherTranscripts != null ? otherTranscripts : "");
    }

    //==================================================================================================================

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        final GencodeFuncotation that = (GencodeFuncotation) o;

        if (start != that.start) return false;
        if (end != that.end) return false;
        if (transcriptExon != that.transcriptExon) return false;
        if (transcriptPos != that.transcriptPos) return false;
        if (hugoSymbol != null ? !hugoSymbol.equals(that.hugoSymbol) : that.hugoSymbol != null) return false;
        if (ncbiBuild != null ? !ncbiBuild.equals(that.ncbiBuild) : that.ncbiBuild != null) return false;
        if (chromosome != null ? !chromosome.equals(that.chromosome) : that.chromosome != null) return false;
        if (variantClassification != that.variantClassification) return false;
        if (variantType != that.variantType) return false;
        if (refAllele != null ? !refAllele.equals(that.refAllele) : that.refAllele != null) return false;
        if (tumorSeqAllele1 != null ? !tumorSeqAllele1.equals(that.tumorSeqAllele1) : that.tumorSeqAllele1 != null)
            return false;
        if (tumorSeqAllele2 != null ? !tumorSeqAllele2.equals(that.tumorSeqAllele2) : that.tumorSeqAllele2 != null)
            return false;
        if (genomeChange != null ? !genomeChange.equals(that.genomeChange) : that.genomeChange != null) return false;
        if (annotationTranscript != null ? !annotationTranscript.equals(that.annotationTranscript) : that.annotationTranscript != null)
            return false;
        if (transcriptStrand != null ? !transcriptStrand.equals(that.transcriptStrand) : that.transcriptStrand != null)
            return false;
        if (cDnaChange != null ? !cDnaChange.equals(that.cDnaChange) : that.cDnaChange != null) return false;
        if (codonChange != null ? !codonChange.equals(that.codonChange) : that.codonChange != null) return false;
        if (proteinChange != null ? !proteinChange.equals(that.proteinChange) : that.proteinChange != null)
            return false;
        return otherTranscripts != null ? otherTranscripts.equals(that.otherTranscripts) : that.otherTranscripts == null;
    }

    @Override
    public int hashCode() {
        int result = hugoSymbol != null ? hugoSymbol.hashCode() : 0;
        result = 31 * result + (ncbiBuild != null ? ncbiBuild.hashCode() : 0);
        result = 31 * result + (chromosome != null ? chromosome.hashCode() : 0);
        result = 31 * result + start;
        result = 31 * result + end;
        result = 31 * result + (variantClassification != null ? variantClassification.hashCode() : 0);
        result = 31 * result + (variantType != null ? variantType.hashCode() : 0);
        result = 31 * result + (refAllele != null ? refAllele.hashCode() : 0);
        result = 31 * result + (tumorSeqAllele1 != null ? tumorSeqAllele1.hashCode() : 0);
        result = 31 * result + (tumorSeqAllele2 != null ? tumorSeqAllele2.hashCode() : 0);
        result = 31 * result + (genomeChange != null ? genomeChange.hashCode() : 0);
        result = 31 * result + (annotationTranscript != null ? annotationTranscript.hashCode() : 0);
        result = 31 * result + (transcriptStrand != null ? transcriptStrand.hashCode() : 0);
        result = 31 * result + transcriptExon;
        result = 31 * result + transcriptPos;
        result = 31 * result + (cDnaChange != null ? cDnaChange.hashCode() : 0);
        result = 31 * result + (codonChange != null ? codonChange.hashCode() : 0);
        result = 31 * result + (proteinChange != null ? proteinChange.hashCode() : 0);
        result = 31 * result + (otherTranscripts != null ? otherTranscripts.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return "GencodeFuncotation{" +
                "hugoSymbol='" + hugoSymbol + '\'' +
                ", ncbiBuild='" + ncbiBuild + '\'' +
                ", chromosome='" + chromosome + '\'' +
                ", start=" + start +
                ", end=" + end +
                ", variantClassification=" + variantClassification +
                ", variantType=" + variantType +
                ", refAllele='" + refAllele + '\'' +
                ", tumorSeqAllele1='" + tumorSeqAllele1 + '\'' +
                ", tumorSeqAllele2='" + tumorSeqAllele2 + '\'' +
                ", genomeChange='" + genomeChange + '\'' +
                ", annotationTranscript='" + annotationTranscript + '\'' +
                ", transcriptStrand='" + transcriptStrand + '\'' +
                ", transcriptExon=" + transcriptExon +
                ", transcriptPos=" + transcriptPos +
                ", cDnaChange='" + cDnaChange + '\'' +
                ", codonChange='" + codonChange + '\'' +
                ", proteinChange='" + proteinChange + '\'' +
                ", otherTranscripts=" + otherTranscripts +
                '}';
    }

    //==================================================================================================================

    public String getHugoSymbol() {
        return hugoSymbol;
    }

    public void setHugoSymbol(final String hugoSymbol) {
        this.hugoSymbol = hugoSymbol;
    }

    public String getNcbiBuild() {
        return ncbiBuild;
    }

    public void setNcbiBuild(final String ncbiBuild) {
        this.ncbiBuild = ncbiBuild;
    }

    public String getChromosome() {
        return chromosome;
    }

    public void setChromosome(final String chromosome) {
        this.chromosome = chromosome;
    }

    public int getStart() {
        return start;
    }

    public void setStart(final int start) {
        this.start = start;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(final int end) {
        this.end = end;
    }

    public VariantClassification getVariantClassification() {
        return variantClassification;
    }

    public void setVariantClassification(final VariantClassification variantClassification) {
        this.variantClassification = variantClassification;
    }

    public VariantType getVariantType() {
        return variantType;
    }

    public void setVariantType(final VariantType variantType) {
        this.variantType = variantType;
    }

    public String getRefAllele() {
        return refAllele;
    }

    public void setRefAllele(final String refAllele) {
        this.refAllele = refAllele;
    }

    public String getTumorSeqAllele1() {
        return tumorSeqAllele1;
    }

    public void setTumorSeqAllele1(final String tumorSeqAllele1) {
        this.tumorSeqAllele1 = tumorSeqAllele1;
    }

    public String getTumorSeqAllele2() {
        return tumorSeqAllele2;
    }

    public void setTumorSeqAllele2(final String tumorSeqAllele2) {
        this.tumorSeqAllele2 = tumorSeqAllele2;
    }

    public String getGenomeChange() {
        return genomeChange;
    }

    public void setGenomeChange(final String genomeChange) {
        this.genomeChange = genomeChange;
    }

    public String getAnnotationTranscript() {
        return annotationTranscript;
    }

    public void setAnnotationTranscript(final String annotationTranscript) {
        this.annotationTranscript = annotationTranscript;
    }

    public String getTranscriptStrand() {
        return transcriptStrand;
    }

    public void setTranscriptStrand(final String transcriptStrand) {
        this.transcriptStrand = transcriptStrand;
    }

    public int getTranscriptExon() {
        return transcriptExon;
    }

    public void setTranscriptExon(final int transcriptExon) {
        this.transcriptExon = transcriptExon;
    }

    public int getTranscriptPos() {
        return transcriptPos;
    }

    public void setTranscriptPos(final int transcriptPos) {
        this.transcriptPos = transcriptPos;
    }

    public String getcDnaChange() {
        return cDnaChange;
    }

    public void setcDnaChange(final String cDnaChange) {
        this.cDnaChange = cDnaChange;
    }

    public String getCodonChange() {
        return codonChange;
    }

    public void setCodonChange(final String codonChange) {
        this.codonChange = codonChange;
    }

    public String getProteinChange() {
        return proteinChange;
    }

    public void setProteinChange(final String proteinChange) {
        this.proteinChange = proteinChange;
    }

    public List<String> getOtherTranscripts() {
        return otherTranscripts;
    }

    public void setOtherTranscripts(final List<String> otherTranscripts) {
        this.otherTranscripts = otherTranscripts;
    }


    //==================================================================================================================

    public enum VariantType {
        INS("INS"),
        DEL("DEL"),
        SNP("SNP"),
        DNP("DNP"),
        TNP("TNP"),
        ONP("ONP"),
        MNP("MNP"),
        xNP("NP");

        final private String serialized;

        VariantType(final String serialized) { this.serialized = serialized; }

        @Override
        public String toString() {
            return serialized;
        }
    }

    /**
     * Represents the type of variant found.
     */
    public enum VariantClassification {
        INTRON(10),
        FIVE_PRIME_UTR(6),
        THREE_PRIME_UTR(6),
        IGR(20),
        FIVE_PRIME_FLANK(15),
        THREE_PRIME_FLANK(15),
        MISSENSE(1),
        NONSENSE(0),
        NONSTOP(0),
        SILENT(5),
        SPLICE_SITE(4),
        IN_FRAME_DEL(1),
        IN_FRAME_INS(1),
        FRAME_SHIFT_INS(2),
        FRAME_SHIFT_DEL(2),
        FRAME_SHIFT_SUB(2),
        START_CODON_SNP(3),
        START_CODON_INS(3),
        START_CODON_DEL(3),
        STOP_CODON_INS(3),
        STOP_CODON_DEL(3),
        STOP_CODON_SNP(3),
        DE_NOVO_START_IN_FRAME(1),
        DE_NOVO_START_OUT_FRAME(0),
        RNA(4),
        LINCRNA(4);

        /**
         * The relative severity of each {@link VariantClassification}.
         * Lower numbers are considered more severe.
         * Higher numbers are considered less severe.
         */
        final private int relativeSeverity;

        VariantClassification(final int sev) {
            relativeSeverity = sev;
        }
    }
}
