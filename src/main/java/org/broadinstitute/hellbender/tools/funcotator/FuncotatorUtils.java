/*
* Copyright 2012-2016 Broad Institute, Inc.
* 
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
* 
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
* THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package org.broadinstitute.hellbender.tools.funcotator;

/*
 * Copyright (c) 2010 The Broad Institute
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

import htsjdk.samtools.reference.ReferenceSequence;
import htsjdk.samtools.util.Locatable;
import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.VariantContext;
import org.apache.log4j.Logger;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class FuncotatorUtils {

    private final static Logger logger = Logger.getLogger(FuncotatorUtils.class);

    /**
     * PRIVATE CONSTRUCTOR
     * DO NOT INSTANTIATE THIS CLASS!
     */
    private FuncotatorUtils() {}

    private static final HashMap<String,AminoAcid> tableByCodon = new HashMap<>(AminoAcid.values().length);
    private static final HashMap<String,AminoAcid> tableByCode = new HashMap<>(AminoAcid.values().length);
    private static final HashMap<String,AminoAcid> tableByLetter = new HashMap<>(AminoAcid.values().length);

    /**
     * Initialize our hashmaps of lookup tables:
     */
    static {
        for ( final AminoAcid acid : AminoAcid.values() ) {
            tableByCode.put(acid.getCode(),acid);
            tableByLetter.put(acid.getLetter(), acid);
            for ( final String codon : acid.codons ) {
                tableByCodon.put(codon,acid);
            }
        }
    }

    /**
     * Returns the {@link AminoAcid} corresponding to the given three-letter Eukaryotic {@code codon}
     * The codons given are expected to be valid for Eukaryotic DNA.
     * @param codon The three-letter codon (each letter one of A,T,G,C) representing a Eukaryotic {@link AminoAcid}
     * @return The {@link AminoAcid} corresponding to the given {@code codon}.  Returns {@code null} if the given {@code codon} does not code for a Eucaryotic {@link AminoAcid}.
     */
    public static AminoAcid getEukaryoticAminoAcidByCodon(final String codon) {
        if (codon == null) {
            return null;
        }
        return tableByCodon.get(codon.toUpperCase());
    }

    /**
     * Returns the {@link AminoAcid} corresponding to the given three-letter Mitochondrial {@code codon}.
     * The codons given are expected to be valid for Mitochondrial DNA.
     * @param codon The three-letter codon (each letter one of A,T,G,C) representing a Mitochondrial {@link AminoAcid}
     * @return The {@link AminoAcid} corresponding to the given {@code codon}.  Returns {@code null} if the given {@code codon} does not code for a Mitochondrial {@link AminoAcid}.
     */
    public static AminoAcid getMitochondrialAminoAcidByCodon(final String codon, final boolean isFirst) {

        if (codon == null) {
            return null;
        }

        final String upperCodon = codon.toUpperCase();
        if ( isFirst && upperCodon.equals("ATT") || upperCodon.equals("ATA") ) {
            return AminoAcid.METHIONINE;
        } else if ( upperCodon.equals("AGA") || upperCodon.equals("AGG") ) {
            return AminoAcid.STOP_CODON;
        } else if ( upperCodon.equals("TGA") ) {
            return AminoAcid.TRYPTOPHAN;
        } else {
            return tableByCodon.get(upperCodon);
        }
    }

    /**
     * Returns the {@link AminoAcid} corresponding to the given single-letter abbreviation.
     * @param letter The one-letter abbreviation representing an {@link AminoAcid}
     * @return The {@link AminoAcid} corresponding to the given {@code letter}.  Returns {@code null} if the given {@code letter} does not code for an {@link AminoAcid}.
     */
    public static AminoAcid getAminoAcidByLetter(final String letter) {
        if ( letter == null ) {
            return null;
        }

        return tableByLetter.get(letter);
    }

    /**
     * Returns the {@link AminoAcid} corresponding to the given single-letter abbreviation.
     * @param letter The one-letter abbreviation representing an {@link AminoAcid}
     * @return The {@link AminoAcid} corresponding to the given {@code letter}.  Returns {@code null} if the given {@code letter} does not code for an {@link AminoAcid}.
     */
    public static AminoAcid getAminoAcidByLetter(final char letter) {
        return tableByLetter.get(String.valueOf(letter));
    }

    /**
     * @return A {@link String} array of long names for all amino acids in {@link AminoAcid}
     */
    public static String[] getAminoAcidNames() {
        final String[] names = new String[AminoAcid.values().length];
        for ( final AminoAcid acid : AminoAcid.values() ) {
            names[acid.ordinal()] = acid.getName();
        }

        return names;
    }

    /**
     * @return A {@link String} array of short names / three-letter abbreviations for all amino acids in {@link AminoAcid}
     */
    public static String[] getAminoAcidCodes() {
        final String[] codes = new String[AminoAcid.values().length];
        for ( final AminoAcid acid : AminoAcid.values() ) {
            codes[acid.ordinal()] = acid.getCode();
        }

        return codes;
    }

    /**
     * Determines whether the given reference and alternate alleles constitute a frameshift mutation.
     * @param reference The reference {@link Allele}.
     * @param alternate The alternate / variant {@link Allele}.
     * @return {@code true} if replacing the reference with the alternate results in a frameshift.  {@code false} otherwise.
     */
    public static boolean isFrameshift(final Allele reference, final Allele alternate) {

        Utils.nonNull(reference);
        Utils.nonNull(alternate);

        // We know it's a frameshift if we have a replacement that is not of a
        // length evenly divisible by 3 because that's how many bases are read at once:
        return ((Math.abs( reference.length() - alternate.length() ) % 3) != 0);
    }

    /**
     * Determines whether the given reference and alternate alleles constitute a frameshift mutation.
     * @param startPos Genomic start position (1-based, inclusive) of the variant.
     * @param refEnd Genomic end position (1-based, inclusive) of the reference allele.
     * @param altEnd Genomic end position (1-based, inclusive) of the alternate allele.
     * @return {@code true} if replacing the reference with the alternate results in a frameshift.  {@code false} otherwise.
     */
    public static boolean isFrameshift(final int startPos, final int refEnd, final int altEnd) {

        final int refLength = refEnd - startPos + 1;
        final int altLength = altEnd - startPos + 1;

        // We know it's a frameshift if we have a replacement that is not of a
        // length evenly divisible by 3 because that's how many bases are read at once:
        return ((Math.abs( refLength - altLength ) % 3) != 0);
    }

    /**
     * Gets the position describing where the given allele and variant lie inside the given transcript using transcript-based coordinates.
     * The index will be calculated even if the given variant ends outside the bounds of the given transcript.
     * @param variant A {@link Locatable} to locate inside the given {@code transcript}.
     * @param transcript A {@link List} of {@link Locatable} that describe the transcript to use for locating the given {@code allele}.
     * @return The position (1-based, inclusive) describing where the given {@code allele} lies in the given {@code transcript}.  If the variant is not in the given {@code transcript}, then this returns -1.
     */
    public static int getStartPositionInTranscript(final Locatable variant,
                                                   final List<? extends Locatable> transcript) {
        Utils.nonNull(variant);
        Utils.nonNull(transcript);

        int position = 0;

        boolean foundPosition = false;

        final SimpleInterval variantStartLocus = new SimpleInterval(variant.getContig(), variant.getStart(), variant.getStart());

        for (final Locatable exon : transcript) {
            if (!exon.getContig().equals(variantStartLocus.getContig())) {
                throw new GATKException("Variant and transcript contigs are not equal: "
                        + variantStartLocus.getContig() + " != " + exon.getContig());
            }

            if (new SimpleInterval(exon).contains(variantStartLocus)) {
                position += variantStartLocus.getStart() - exon.getStart();
                foundPosition = true;
                break;
            } else {
                // Add 1 because of inclusive positions / indexing starting at 1
                position += exon.getEnd() - exon.getStart() + 1;
            }
        }

        if ( foundPosition ) {
            return position + 1;
        }

        return -1;
    }

    /**
     * Get the sequence-aligned end position for the given allele and start position.
     * @param seqAlignedStart The sequence-aligned starting position (1-based, inclusive) from which to calculate the end position.
     * @param alleleLength The length of the allele in bases.
     * @return An aligned end position (1-based, inclusive) for the given codon start and allele length.
     */
    public static int getAlignedEndPosition(final int seqAlignedStart, final int alleleLength) {
        // We subtract 1 because the start and end positions must be inclusive.
        return (int)(Math.ceil((seqAlignedStart + alleleLength - 1) / 3.0) * 3);
    }

    /**
     * Gets the sequence aligned position (1-based, inclusive) for the given coding sequence position.
     * This will produce the next lowest position evenly divisible by 3, such that a codon starting at this returned
     * position would include the given position.
     * @param position A sequence starting coordinate for which to produce an coding-aligned position.
     * @return A coding-aligned position (1-based, inclusive) corresponding to the given {@code position}
     */
    public static int getAlignedPosition(final int position) {
        return position - ((position - 1) % 3);
    }

    /**
     * Calculates whether the given {@code startPosition} (1-based, inclusive) is in frame relative to the end of the region.
     * @param startPosition The position (1-based, inclusive) relative to the start of a region to check for frame alignment.
     * @param regionLength The length of the region containing {@code startPosition}.
     * @return {@code true} if the given {@code startPosition} is in frame relative to the given {@code regionLength} ; {@code false} otherwise.
     */
    public static boolean isInFrameWithEndOfRegion(final int startPosition, final int regionLength) {
        return (((regionLength - startPosition + 1) % 3) == 0);
    }

    /**
     * Creates the string representation of the codon change for the given {@link SequenceComparison}.
     * @param seqComp {@link SequenceComparison} representing the alternate and reference alleles for a DNA sequence.
     * @return A {@link String} representing the codon change for the given {@link SequenceComparison}.
     */
    public static String getCodonChangeString(final SequenceComparison seqComp) {

        Utils.nonNull(seqComp);
        Utils.nonNull(seqComp.getAlignedCodingSequenceAlleleStart());
        Utils.nonNull(seqComp.getAlignedReferenceAlleleStop());
        Utils.nonNull(seqComp.getAlignedReferenceAllele());
        Utils.nonNull(seqComp.getAlignedAlternateAllele());
        Utils.nonNull(seqComp.getCodingSequenceAlleleStart());
        
        final String ref = seqComp.getAlignedReferenceAllele();
        final String alt = seqComp.getAlignedAlternateAllele();

        if ( seqComp.getAlignedCodingSequenceAlleleStart() == seqComp.getAlignedReferenceAlleleStop() ) {
            return "c.(" + seqComp.getAlignedCodingSequenceAlleleStart() + ")" +
                    ref + ">" + alt;
        }
        else {
            return "c.(" + seqComp.getAlignedCodingSequenceAlleleStart() + "-" +
                    seqComp.getAlignedReferenceAlleleStop() + ")" +
                    ref + ">" + alt;
        }
    }

    /**
     * Creates the string representation of the codon change for the given {@link SequenceComparison}.
     * @param seqComp {@link SequenceComparison} representing the alternate and reference alleles for a DNA sequence.
     * @return A {@link String} representing the codon change for the given {@link SequenceComparison}.
     */
    public static String getProteinChangeString(final SequenceComparison seqComp) {

        Utils.nonNull(seqComp);
        Utils.nonNull(seqComp.getReferenceAminoAcidSequence());
        Utils.nonNull(seqComp.getProteinChangeStartPosition());
        Utils.nonNull(seqComp.getProteinChangeEndPosition());
        Utils.nonNull(seqComp.getAlternateAminoAcidSequence());

        if ( seqComp.getProteinChangeStartPosition().equals(seqComp.getProteinChangeEndPosition()) ) {
            return "p." + seqComp.getReferenceAminoAcidSequence() + seqComp.getProteinChangeStartPosition() +
                    seqComp.getAlternateAminoAcidSequence();
        }
        else {
            return "p." + seqComp.getReferenceAminoAcidSequence() + seqComp.getProteinChangeStartPosition()
                    + "-" + seqComp.getProteinChangeEndPosition() + seqComp.getAlternateAminoAcidSequence();
        }
    }

    /**
     * Get the coding sequence change string from the given {@link SequenceComparison}
     * @param seqComp {@link SequenceComparison} from which to construct the coding sequence change string.
     * @return A {@link String} representing the coding sequence change between the ref and alt alleles in {@code seqComp}.
     */
    public static String getCodingSequenceChangeString( final SequenceComparison seqComp ) {

        Utils.nonNull(seqComp);
        Utils.nonNull(seqComp.getCodingSequenceAlleleStart());
        Utils.nonNull(seqComp.getReferenceAminoAcidSequence());
        Utils.nonNull(seqComp.getAlternateAminoAcidSequence());

        return "c." + seqComp.getCodingSequenceAlleleStart() +
                seqComp.getReferenceAllele() + ">" + seqComp.getAlternateAllele();
    }

    /**
     * Creates an amino acid sequence from a given coding sequence.
     * If the coding sequence is not evenly divisible by 3, the remainder bases will not be included in the coding sequence.
     * @param codingSequence The coding sequence from which to create an amino acid sequence.
     * @return A {@link String} containing a sequence of single-letter amino acids.
     */
    public static String createAminoAcidSequence(final String codingSequence) {

        Utils.nonNull(codingSequence);

        final StringBuilder sb = new StringBuilder();

        // Ensure that we don't have remainder bases:
        int maxIndex = codingSequence.length();
        if ( maxIndex % 3 != 0 ) {
            maxIndex = (int)Math.floor(maxIndex / 3) * 3;
            logger.warn("createAminoAcidSequence given a coding sequence of length not divisible by 3.  Dropping bases from the end: " + (codingSequence.length() % 3));
        }

        for ( int i = 0; i < maxIndex; i += 3 ) {
            final AminoAcid aa = getEukaryoticAminoAcidByCodon(codingSequence.substring(i, i+3));
            if ( aa == null ) {
                sb.append(AminoAcid.NONSENSE.getLetter());
            }
            else {
                sb.append(aa.getLetter());
            }
        }
        return sb.toString();
    }

    /**
     * Get the full alternate coding sequence given a reference coding sequence, and two alleles.
     * @param referenceCodingSequence The reference sequence on which to base the resulting alternate coding sequence.
     * @param alleleStartPos Starting position (1-based, inclusive) for the ref and alt alleles in the given {@code referenceCodingSequence}.
     * @param refAllele Reference Allele.
     * @param altAllele Alternate Allele.
     * @return The coding sequence that includes the given alternate allele in place of the given reference allele.
     */
    public static String getAlternateCodingSequence( final String referenceCodingSequence, final int alleleStartPos,
                                                     final Allele refAllele, final Allele altAllele ) {

        Utils.nonNull(referenceCodingSequence);
        Utils.nonNull(refAllele);
        Utils.nonNull(altAllele);

        // We have to subtract 1 here because we need to account for the 1-based indexing of
        // the start and end of the coding region:
        return referenceCodingSequence.substring(0, alleleStartPos - 1) +
                altAllele.getBaseString() +
                referenceCodingSequence.substring(alleleStartPos - 1 + refAllele.length());
    }

    /**
     * Creates and returns the coding sequence given a {@link ReferenceContext} and a {@link List} of {@link Locatable} representing a set of Exons.
     * Locatables start and end values are inclusive.
     * Assumes {@code exonList} ranges are indexed by 1.
     * @param reference A {@link ReferenceContext} from which to construct the coding region.
     * @param exonList A {@link List} of {@link Locatable} representing a set of Exons to be concatenated together to create the coding sequence.
     * @return A string of bases for the given {@code exonList} concatenated together.
     */
    public static String getCodingSequence(final ReferenceContext reference, final List<? extends Locatable> exonList) {

        Utils.nonNull(reference);
        Utils.nonNull(exonList);

        // Sanity check:
        if (exonList.size() == 0) {
            return "";
        }

        final StringBuilder sb = new StringBuilder();

        int start = Integer.MAX_VALUE;
        int end = Integer.MIN_VALUE;

        // Start by sorting our list of exons.
        // This is very important to ensure that we have all sequences in the right order at the end
        // and so we can support different read directions:
        exonList.sort((lhs, rhs) -> lhs.getStart() < rhs.getStart() ? -1 : (lhs.getStart() > rhs.getStart() ) ? 1 : 0 );

        for ( final Locatable exon : exonList ) {

            // First a basic sanity check:
            if ( !exon.getContig().equals(reference.getWindow().getContig()) ) {
                throw new GATKException("Cannot create a coding sequence! Contigs not the same - Ref: "
                        + reference.getInterval().getContig() + ", Exon: " + exon.getContig());
            }

            if ( start > exon.getStart() ) { start = exon.getStart(); }
            if ( end < exon.getEnd() ) { end = exon.getEnd(); }
        }

        // Set the window on our reference to be correct for our start and end:
        reference.setWindow(
                Math.abs(start - reference.getInterval().getStart()),
                Math.abs(reference.getInterval().getEnd() - end)
        );

        // Now that the window size is correct, we can go through and pull our sequences out.

        // Get the window so we can convert to reference coordinates from genomic coordinates of the exons:
        final SimpleInterval refWindow = reference.getWindow();
        final byte[] bases = reference.getBases();

        // Go through and grab our sequences based on our exons:
        for ( final Locatable exon : exonList ) {

            // Subtract 1 from start because positions are indexed by 1.
            int exonStartArrayCoord = exon.getStart() - refWindow.getStart() - 1;

            // Sanity check just in case the exon and ref window start at the same place:
            if ( exonStartArrayCoord == -1 ) {
                exonStartArrayCoord = 0;
            }

            // Add 1 to end because end range in copyOfRange is exclusive
            final int exonEndArrayCoord = exonStartArrayCoord + (exon.getEnd() - exon.getStart()) + 1;

            // TODO: find a better / faster way to do this:
            sb.append(
                    new String(
                            Arrays.copyOfRange(bases, exonStartArrayCoord, exonEndArrayCoord)
                    )
            );
        }

        return sb.toString();
    }

    /**
     * A simple data object to hold a comparison between a reference sequence and an alternate allele.
     */
    public static class SequenceComparison {
        private ReferenceSequence wholeReferenceSequence = null;

        private String  contig                           = null;
        private Integer alleleStart                      = null;
        private Integer transcriptAlleleStart            = null;
        private Integer codingSequenceAlleleStart        = null;
        private Integer alignedCodingSequenceAlleleStart = null;

        private Integer proteinChangeStartPosition       = null;
        private Integer proteinChangeEndPosition         = null;

        private String referenceAllele                   = null;
        private String alignedReferenceAllele            = null;
        private Integer alignedReferenceAlleleStop       = null;
        private String referenceAminoAcidSequence        = null;

        private String alternateAllele                   = null;
        private String alignedAlternateAllele            = null;
        private Integer alignedAlternateAlleleStop       = null;
        private String alternateAminoAcidSequence        = null;

        // =============================================================================================================

        public ReferenceSequence getWholeReferenceSequence() {
            return wholeReferenceSequence;
        }

        public void setWholeReferenceSequence(final ReferenceSequence wholeReferenceSequence) {
            this.wholeReferenceSequence = wholeReferenceSequence;
        }

        public String getContig() {
            return contig;
        }

        public void setContig(final String contig) {
            this.contig = contig;
        }

        public Integer getAlleleStart() {
            return alleleStart;
        }

        public void setAlleleStart(final Integer alleleStart) {
            this.alleleStart = alleleStart;
        }

        public Integer getTranscriptAlleleStart() {
            return transcriptAlleleStart;
        }

        public void setTranscriptAlleleStart(final Integer transcriptAlleleStart) {
            this.transcriptAlleleStart = transcriptAlleleStart;
        }

        public Integer getCodingSequenceAlleleStart() {
            return codingSequenceAlleleStart;
        }

        public void setCodingSequenceAlleleStart(final Integer codingSequenceAlleleStart) {
            this.codingSequenceAlleleStart = codingSequenceAlleleStart;
        }

        public Integer getAlignedCodingSequenceAlleleStart() {
            return alignedCodingSequenceAlleleStart;
        }

        public void setAlignedCodingSequenceAlleleStart(final Integer alignedCodingSequenceAlleleStart) {
            this.alignedCodingSequenceAlleleStart = alignedCodingSequenceAlleleStart;
        }

        public Integer getProteinChangeStartPosition() {
            return proteinChangeStartPosition;
        }

        public void setProteinChangeStartPosition(final Integer proteinChangeStartPosition) {
            this.proteinChangeStartPosition = proteinChangeStartPosition;
        }

        public Integer getProteinChangeEndPosition() {
            return proteinChangeEndPosition;
        }

        public void setProteinChangeEndPosition(final Integer proteinChangeEndPosition) {
            this.proteinChangeEndPosition = proteinChangeEndPosition;
        }

        public String getReferenceAllele() {
            return referenceAllele;
        }

        public void setReferenceAllele(final String referenceAllele) {
            this.referenceAllele = referenceAllele;
        }

        public String getAlignedReferenceAllele() {
            return alignedReferenceAllele;
        }

        public void setAlignedReferenceAllele(final String alignedReferenceAllele) {
            this.alignedReferenceAllele = alignedReferenceAllele;
        }

        public Integer getAlignedReferenceAlleleStop() {
            return alignedReferenceAlleleStop;
        }

        public void setAlignedReferenceAlleleStop(final Integer alignedReferenceAlleleStop) {
            this.alignedReferenceAlleleStop = alignedReferenceAlleleStop;
        }

        public String getReferenceAminoAcidSequence() {
            return referenceAminoAcidSequence;
        }

        public void setReferenceAminoAcidSequence(final String referenceAminoAcidSequence) {
            this.referenceAminoAcidSequence = referenceAminoAcidSequence;
        }

        public String getAlternateAllele() {
            return alternateAllele;
        }

        public void setAlternateAllele(final String alternateAllele) {
            this.alternateAllele = alternateAllele;
        }

        public String getAlignedAlternateAllele() {
            return alignedAlternateAllele;
        }

        public void setAlignedAlternateAllele(final String alignedAlternateAllele) {
            this.alignedAlternateAllele = alignedAlternateAllele;
        }

        public Integer getAlignedAlternateAlleleStop() {
            return alignedAlternateAlleleStop;
        }

        public void setAlignedAlternateAlleleStop(final Integer alignedAlternateAlleleStop) {
            this.alignedAlternateAlleleStop = alignedAlternateAlleleStop;
        }

        public String getAlternateAminoAcidSequence() {
            return alternateAminoAcidSequence;
        }

        public void setAlternateAminoAcidSequence(final String alternateAminoAcidSequence) {
            this.alternateAminoAcidSequence = alternateAminoAcidSequence;
        }
    }
}
