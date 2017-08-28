package org.broadinstitute.hellbender.tools.funcotator;

import org.broadinstitute.hellbender.utils.BaseUtils;

import java.util.List;

/**
 * A class to represent a Functional Annotation.
 * Created by jonn on 8/22/17.
 */
public class Funcotation {

    private static final String FIELD_DELIMITER = "|";

    //==================================================================================================================

    private String                  allele;
    private VariantClassification   classification;
    private String                  symbol;
    private String                  gene;
    private String                  featureType;
    private String                  feature;
    private String                  biotype;
    private int                     exonNumber;
    private int                     cdsPosition;
    private int                     proteinPosition;

    //==================================================================================================================

    /**
     * Converts this {@link Funcotation} to a string suitable for insertion into a VCF file.
     * @return a {@link String} representing this {@link Funcotation} suitable for insertion into a VCF file.
     */
    public String serializeToVcfString() {

        // Only serialize this funcotation if we have an allele:
        if ( allele == null ) {
            return "";
        }

        return  (allele != null ? allele : "") + FIELD_DELIMITER +
                (classification != null ? classification : "") + FIELD_DELIMITER +
                (symbol != null ? symbol : "") + FIELD_DELIMITER +
                (gene != null ? gene : "") + FIELD_DELIMITER +
                (featureType != null ? featureType : "") + FIELD_DELIMITER +
                (feature != null ? feature : "") + FIELD_DELIMITER +
                (biotype != null ? biotype : "") + FIELD_DELIMITER +
                Integer.toString(exonNumber) + FIELD_DELIMITER +
                Integer.toString(cdsPosition) + FIELD_DELIMITER +
                Integer.toString(proteinPosition);
    }

    //==================================================================================================================

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        final Funcotation that = (Funcotation) o;

        if (cdsPosition != that.cdsPosition) return false;
        if ( exonNumber != that.exonNumber) return false;
        if (proteinPosition != that.proteinPosition) return false;
        if (allele != null ? !allele.equals(that.allele) : that.allele != null) return false;
        if (classification != that.classification) return false;
        if (symbol != null ? !symbol.equals(that.symbol) : that.symbol != null) return false;
        if (gene != null ? !gene.equals(that.gene) : that.gene != null) return false;
        if (featureType != null ? !featureType.equals(that.featureType) : that.featureType != null) return false;
        if (feature != null ? !feature.equals(that.feature) : that.feature != null) return false;
        return biotype != null ? !biotype.equals(that.biotype) : that.biotype != null;
    }

    @Override
    public int hashCode() {
        int result = allele != null ? allele.hashCode() : 0;
        result = 31 * result + (classification != null ? classification.hashCode() : 0);
        result = 31 * result + (symbol != null ? symbol.hashCode() : 0);
        result = 31 * result + (gene != null ? gene.hashCode() : 0);
        result = 31 * result + (featureType != null ? featureType.hashCode() : 0);
        result = 31 * result + (feature != null ? feature.hashCode() : 0);
        result = 31 * result + (biotype != null ? biotype.hashCode() : 0);
        result = 31 * result + exonNumber;
        result = 31 * result + cdsPosition;
        result = 31 * result + proteinPosition;
        return result;
    }

    @Override
    public String toString() {
        return "Funcotation{" +
                "allele=" + allele +
                ", classification=" + classification +
                ", symbol='" + symbol + '\'' +
                ", gene='" + gene + '\'' +
                ", featureType='" + featureType + '\'' +
                ", feature='" + feature + '\'' +
                ", biotype='" + biotype + '\'' +
                ", exon='" + exonNumber + '\'' +
                ", cdsPosition=" + cdsPosition +
                ", proteinPosition=" + proteinPosition +
                '}';
    }

    //==================================================================================================================

    public String getAllele() {
        return allele;
    }

    public void setAllele(final String allele) {
        this.allele = allele;
    }

    public VariantClassification getClassification() {
        return classification;
    }

    public void setClassification(final VariantClassification classification) {
        this.classification = classification;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(final String symbol) {
        this.symbol = symbol;
    }

    public String getGene() {
        return gene;
    }

    public void setGene(final String gene) {
        this.gene = gene;
    }

    public String getFeatureType() {
        return featureType;
    }

    public void setFeatureType(final String featureType) {
        this.featureType = featureType;
    }

    public String getFeature() {
        return feature;
    }

    public void setFeature(final String feature) {
        this.feature = feature;
    }

    public String getBiotype() {
        return biotype;
    }

    public void setBiotype(final String biotype) {
        this.biotype = biotype;
    }

    public int getExon() {
        return exonNumber;
    }

    public void setExon(final int exonNumber) {
        this.exonNumber = exonNumber;
    }

    public int getCdsPosition() {
        return cdsPosition;
    }

    public void setCdsPosition(final int cdsPosition) {
        this.cdsPosition = cdsPosition;
    }

    public int getProteinPosition() {
        return proteinPosition;
    }

    public void setProteinPosition(final int proteinPosition) {
        this.proteinPosition = proteinPosition;
    }


    //==================================================================================================================

    /**
     * Represents the type of variant found.
     */
    enum VariantClassification {
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
