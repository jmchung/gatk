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

import java.util.HashMap;

public class AminoAcidUtils {

    private static final HashMap<String,AminoAcid> tableByCodon = new HashMap<String,AminoAcid>(AminoAcid.values().length);
    private static final HashMap<String,AminoAcid> tableByCode = new HashMap<String,AminoAcid>(AminoAcid.values().length);

    /**
     * Initialize our hashmaps of lookup tables:
     */
    static {
        for ( final AminoAcid acid : AminoAcid.values() ) {
            tableByCode.put(acid.getCode(),acid);
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
        return tableByCodon.get(codon.toUpperCase());
    }

    /**
     * Returns the {@link AminoAcid} corresponding to the given three-letter Mitochondrial {@code codon}.
     * The codons given are expected to be valid for Mitochondrial DNA.
     * @param codon The three-letter codon (each letter one of A,T,G,C) representing a Mitochondrial {@link AminoAcid}
     * @return The {@link AminoAcid} corresponding to the given {@code codon}.  Returns {@code null} if the given {@code codon} does not code for a Mitochondrial {@link AminoAcid}.
     */
    public static AminoAcid getMitochondrialAminoAcidByCodon(final String codon, final boolean isFirst) {
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
     * Gets the {@link AminoAcid} corresponding to the given three-letter abbreviation, {@code code}
     * @param code a three-letter {@link String} corresponding to an {@link AminoAcid}
     * @return The {@link AminoAcid} corresponding to the given {@code code}.  Returns {@code null} if the given {@code code} is not a valid {@link AminoAcid}.
     */
    public static AminoAcid getAminoAcidByCode(final String code) {
        return tableByCode.get(code);
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
}
