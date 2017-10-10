package org.broadinstitute.hellbender.tools.funcotator.dataSources.gencode;

import java.util.Arrays;
import java.util.List;

/**
 * A simple holder class for MNP data providers for the PIK3CA gene.
 * These cases were pulled directly from Oncotator (MUC16ChangeTestdata.py).
 * Created by jonn on 9/21/17.
 */
public class DataProviderForPik3caMnpFullData {
    public static List<Object[]> providePik3caMnpData() {
        return Arrays.asList(
                new Object[]{"PIK3CA", 3, 178916619, 178916619, GencodeFuncotation.VariantClassification.SILENT, GencodeFuncotation.VariantType.SNP, "T", "A", "g.chr3:178916619T>A", "+", "c.6T>A", "c.(4-6)ccT>ccA", "p.P2P"},
                new Object[]{"PIK3CA", 3, 178916620, 178916620, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178916620C>T", "+", "c.7C>T", "c.(7-9)Cca>Tca", "p.P3S"},
                new Object[]{"PIK3CA", 3, 178916617, 178916617, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178916617C>T", "+", "c.4C>T", "c.(4-6)Cct>Tct", "p.P2S"},
                new Object[]{"PIK3CA", 3, 178919220, 178919220, GencodeFuncotation.VariantClassification.SILENT, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178919220C>T", "+", "c.705C>T", "c.(703-705)tcC>tcT", "p.S235S"},
                new Object[]{"PIK3CA", 3, 178921433, 178921433, GencodeFuncotation.VariantClassification.SILENT, GencodeFuncotation.VariantType.SNP, "A", "T", "g.chr3:178921433A>T", "+", "c.915A>T", "c.(913-915)ccA>ccT", "p.P305P"},
                new Object[]{"PIK3CA", 3, 178922366, 178922366, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "T", "A", "g.chr3:178922366T>A", "+", "c.1135T>A", "c.(1135-1137)Tcc>Acc", "p.S379T"},
                new Object[]{"PIK3CA", 3, 178928317, 178928317, GencodeFuncotation.VariantClassification.SILENT, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178928317C>T", "+", "c.1503C>T", "c.(1501-1503)tcC>tcT", "p.S501S"},
                new Object[]{"PIK3CA", 3, 178936091, 178936091, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "G", "A", "g.chr3:178936091G>A", "+", "c.1633G>A", "c.(1633-1635)Gag>Aag", "p.E545K"},
                new Object[]{"PIK3CA", 3, 178937063, 178937063, GencodeFuncotation.VariantClassification.NONSENSE, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178937063C>T", "+", "c.1744C>T", "c.(1744-1746)Cag>Tag", "p.Q582*"},
                new Object[]{"PIK3CA", 3, 178941890, 178941890, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "G", "A", "g.chr3:178941890G>A", "+", "c.2209G>A", "c.(2209-2211)Gag>Aag", "p.E737K"},
                new Object[]{"PIK3CA", 3, 178942511, 178942511, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178942511C>T", "+", "c.2318C>T", "c.(2317-2319)tCc>tTc", "p.S773F"},
                new Object[]{"PIK3CA", 3, 178942523, 178942523, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "G", "A", "g.chr3:178942523G>A", "+", "c.2330G>A", "c.(2329-2331)aGg>aAg", "p.R777K"},
                new Object[]{"PIK3CA", 3, 178943785, 178943785, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "C", "T", "g.chr3:178943785C>T", "+", "c.2452C>T", "c.(2452-2454)Cgt>Tgt", "p.R818C"},
                new Object[]{"PIK3CA", 3, 178947158, 178947158, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "G", "A", "g.chr3:178947158G>A", "+", "c.2594G>A", "c.(2593-2595)gGc>gAc", "p.G865D"},
                new Object[]{"PIK3CA", 3, 178952085, 178952085, GencodeFuncotation.VariantClassification.MISSENSE, GencodeFuncotation.VariantType.SNP, "A", "T", "g.chr3:178952085A>T", "+", "c.3140A>T", "c.(3139-3141)cAt>cTt", "p.H1047L"}
        );
    }

    /**
     * @return Test data for PIK3CA INDELs as taken from Oncotator:VariantClassifierTest.py:test_pik3ca_change_transcript:203-222
     */
    public static List<Object[]> providePik3caInDelData() {
        return Arrays.asList(
            // TODO: FROM ONCOTATOR: Issue 174... uncomment the next line for a unit test that will fail protein change (at least)
            // new Object[] {"PIK3CA", 3, 178948160, 178948168", GencodeFuncotation.VariantClassificaton.SPLICE_SITE, "DEL", "GAGAGGTGA", "-", "g.chr3:178948160_178948164delGAGAG", "+", "c.2936_splice",  "c.e20+1", "p.ER978fs"}, # This is actually a frame shift (p.ER978fs?), since 5 codon bases are deleted
            new Object[] {"PIK3CA", 3, 178916619, 178916620, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",        "CG",    "g.chr3:178916619_178916620insCG",    "+", "c.6_7insCG",          "c.(7-9)ccafs",            "p.P3fs"},
            new Object[] {"PIK3CA", 3, 178916619, 178916620, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",        "CGA",   "g.chr3:178916619_178916620insCGA",   "+", "c.6_7insCGA",         "c.(7-9)cca>CGAcca",       "p.2_3insR"},
            new Object[] {"PIK3CA", 3, 178948154, 178948155, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",        "GAATT", "g.chr3:178948154_178948155insGAATT", "+", "c.2926_2930insGAATT", "c.(2926-2928)gaafs",      "p.E976fs"},
            new Object[] {"PIK3CA", 3, 178948155, 178948156, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",        "GAATT", "g.chr3:178948155_178948156insGAATT", "+", "c.2927_2931insGAATT", "c.(2926-2931)gaatttfs",   "p.F977fs"},  // issue 109 is around this entry
            new Object[] {"PIK3CA", 3, 178948159, 178948160, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",        "GA",    "g.chr3:178948159_178948160insGA",    "+", "c.2931_2932insGA",    "c.(2932-2934)gagfs",      "p.E978fs"},
            new Object[] {"PIK3CA", 3, 178916938, 178916940, GencodeFuncotation.VariantClassification.IN_FRAME_DEL,    GencodeFuncotation.VariantType.DEL, "GAA",      "-",     "g.chr3:178916938_178916940delGAA",   "+", "c.325_327delGAA",     "c.(325-327)gaadel",       "p.E110del"},
            new Object[] {"PIK3CA", 3, 178948159, 178948160, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",        "GAG",   "g.chr3:178948159_178948160insGAG",   "+", "c.2931_2932insGAG",   "c.(2932-2934)gag>GAGgag", "p.978_978E>EE"},
            new Object[] {"PIK3CA", 3, 178948160, 178948162, GencodeFuncotation.VariantClassification.IN_FRAME_DEL,    GencodeFuncotation.VariantType.DEL, "GAG",      "-",     "g.chr3:178948160_178948162delGAG",   "+", "c.2932_2934delGAG",   "c.(2932-2934)gagdel",     "p.E978del"},
            new Object[] {"PIK3CA", 3, 178948160, 178948161, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "GA",       "-",     "g.chr3:178948160_178948161delGA",    "+", "c.2932_2933delGA",    "c.(2932-2934)gagfs",      "p.E978fs"},
            new Object[] {"PIK3CA", 3, 178948160, 178948164, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.DEL, "GAGAG",    "-",     "g.chr3:178948160_178948164delGAGAG", "+", "c.2936_splice",       "c.(2932-2937)gagagg>g",   "p.ER978fs"},
            new Object[] {"PIK3CA", 3, 178948160, 178948167, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.DEL, "GAGAGGTG", "-",     "g.chr3:178948160_178948164delGAGAG", "+", "c.2936_splice",       "c.(2932-2937)gagagg>g",   "p.ER978fs"},
            new Object[] {"PIK3CA", 3, 178948166, 178948168, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.DEL, "TGA",      "-",     "g.chr3:178948166_178948168delTGA",   "+", "c.2936_splice",       "c.e20+2",                 ""},
            new Object[] {"PIK3CA", 3, 178948163, 178948164, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.INS, "-",        "TGA",   "g.chr3:178948163_178948164insTGA",   "+", "c.2936_splice",       "c.(2935-2937)agg>aTGAgg", "p.978_979insM"},
            new Object[] {"PIK3CA", 3, 178948163, 178948164, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.INS, "-",        "T",     "g.chr3:178948163_178948164insT",     "+", "c.2936_splice",       "c.(2935-2937)agg>aTgg",   "p.R979fs"},
            new Object[] {"PIK3CA", 3, 178948165, 178948166, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.INS, "-",        "T",     "g.chr3:178948165_178948166insT",     "+", "c.2936_splice",       "c.e20+1",                 ""},
            new Object[] {"PIK3CA", 3, 178948166, 178948167, GencodeFuncotation.VariantClassification.SPLICE_SITE,     GencodeFuncotation.VariantType.INS, "-",        "T",     "g.chr3:178948166_178948167insT",     "+", "c.2936_splice",       "c.e20+2",                 ""}, // Should this be a splice site?  Technically, this was an insertion of a base 3 away from splice...
            new Object[] {"PIK3CA", 3, 178948154, 178948158, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "GAATT",    "-",     "g.chr3:178948154_178948158delGAATT", "+", "c.2926_2930delGAATT", "c.(2926-2931)gaatttfs",   "p.EF976fs"},
            new Object[] {"PIK3CA", 3, 178948154, 178948157, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "GAAT",     "-",     "g.chr3:178948154_178948158delGAAT",  "+", "c.2926_2929delGAAT",  "c.(2926-2931)gaatttfs",   "p.EF976fs"}
        );
    }

    /**
     * @return Test data for PIK3CA INDELs as taken from Oncotator:VariantClassifierTest.py:test_reference_sequence_codon_construction_positive_strand:510-554
     */
    public static List<Object[]> providePik3caInDelData2() {

        // TODO: This set of tests is missing genome change and cdna change.  These must be added before this method can be used.

        return Arrays.asList(
            // In Frame
            new Object[] {3, 178916621,178916622, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",       "TAT",       "+", "c.(7-12)ccacga>ccTATacga",  "p.3_4PR>PIR"},
            new Object[] {3, 178916622,178916623, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",       "TAT",       "+", "c.(10-12)cga>TATcga",       "p.3_4insY"},
            new Object[] {3, 178916619,178916620, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",       "TAT",       "+", "c.(7-9)cca>TATcca",         "p.2_3insY"},
            new Object[] {3, 178916620,178916621, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",       "TAT",       "+", "c.(7-9)cca>cTATca",         "p.3_3P>LS"},
            new Object[] {3, 178916622,178916623, GencodeFuncotation.VariantClassification.IN_FRAME_INS,    GencodeFuncotation.VariantType.INS, "-",       "CTTGAAGAA", "+", "c.(10-12)cga>CTTGAAGAAcga", "p.3_4insLEE"},
            //fs
            new Object[] {3, 178916621,178916622, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TA",        "+", "c.(7-12)ccacgafs",          "p.R4fs"},
            new Object[] {3, 178916622,178916623, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TA",        "+", "c.(10-12)cgafs",            "p.R4fs"},
            new Object[] {3, 178916619,178916620, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TA",        "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916620,178916621, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TA",        "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916621,178916622, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "T",         "+", "c.(7-12)ccacgafs",          "p.R4fs"},
            new Object[] {3, 178916622,178916623, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "T",         "+", "c.(10-12)cgafs",            "p.R4fs"},
            new Object[] {3, 178916619,178916620, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "T",         "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916620,178916621, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "T",         "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916621,178916622, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TATT",      "+", "c.(7-12)ccacgafs",          "p.R4fs"},
            new Object[] {3, 178916622,178916623, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TATT",      "+", "c.(10-12)cgafs",            "p.R4fs"},
            new Object[] {3, 178916619,178916620, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TATT",      "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916620,178916621, GencodeFuncotation.VariantClassification.FRAME_SHIFT_INS, GencodeFuncotation.VariantType.INS, "-",       "TATT",      "+", "c.(7-9)ccafs",              "p.P3fs"}, // 17

            // DEL
            // In Frame
            new Object[] {3, 178916619, 178916621,GencodeFuncotation.VariantClassification.IN_FRAME_DEL,    GencodeFuncotation.VariantType.DEL, "TCC",     "-",         "+", "c.(4-9)cctcca>cca",         "p.2_3PP>P"},
            new Object[] {3, 178916620, 178916622,GencodeFuncotation.VariantClassification.IN_FRAME_DEL,    GencodeFuncotation.VariantType.DEL, "CCA",     "-",         "+", "c.(7-9)ccadel",             "p.P3del"},
            new Object[] {3, 178916621, 178916623,GencodeFuncotation.VariantClassification.IN_FRAME_DEL,    GencodeFuncotation.VariantType.DEL, "CAC",     "-",         "+", "c.(7-12)ccacga>cga",        "p.P3del"},
            new Object[] {3, 178916622, 178916624,GencodeFuncotation.VariantClassification.IN_FRAME_DEL,    GencodeFuncotation.VariantType.DEL, "ACG",     "-",         "+", "c.(7-12)ccacga>cca",        "p.R4del"}, //21

            //fs
            new Object[] {3, 178916619,178916620, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "TC",      "-",         "+", "c.(4-9)cctccafs",           "p.PP2fs"},
            new Object[] {3, 178916620,178916621, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "CC",      "-",         "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916621,178916622, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "CA",      "-",         "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916622,178916623, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "AC",      "-",         "+", "c.(7-12)ccacgafs",          "p.R4fs"}, // This is correct, since the first amino acid remains the same
            new Object[] {3, 178916619,178916619, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "T",       "-",         "+", "c.(4-6)cctfs",              "p.P3fs"}, //26 -- correct as written, since there are two P's in a row
            new Object[] {3, 178916620,178916620, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "C",       "-",         "+", "c.(7-9)ccafs",              "p.P3fs"},
            new Object[] {3, 178916621,178916621, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "C",       "-",         "+", "c.(7-9)ccafs",              "p.P3fs"}, //28
            new Object[] {3, 178916622,178916622, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "A",       "-",         "+", "c.(7-9)ccafs",              "p.P3fs"}, //29

            new Object[] {3, 178916619,178916622, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "TCCA",    "-",         "+", "c.(4-9)cctccafs",           "p.PP2fs"},
            new Object[] {3, 178916620,178916623, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "CCAC",    "-",         "+", "c.(7-12)ccacgafs",          "p.PR3fs"},
            new Object[] {3, 178916621,178916624, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "CACG",    "-",         "+", "c.(7-12)ccacgafs",          "p.PR3fs"},
            new Object[] {3, 178916622,178916625, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "ACGA",    "-",         "+", "c.(7-12)ccacgafs",          "p.PR3fs"},

            new Object[] {3, 178916619,178916625, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "TCCACGA", "-",         "+", "c.(4-12)cctccacgafs",       "p.PPR2fs"},
            new Object[] {3, 178916620,178916626, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "CCACGAC", "-",         "+", "c.(7-15)ccacgaccafs",       "p.PRP3fs"},
            new Object[] {3, 178916621,178916627, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "CACGACC", "-",         "+", "c.(7-15)ccacgaccafs",       "p.PRP3fs"}, //36
            new Object[] {3, 178916622,178916628, GencodeFuncotation.VariantClassification.FRAME_SHIFT_DEL, GencodeFuncotation.VariantType.DEL, "ACGACCA", "-",         "+", "c.(7-15)ccacgaccafs",       "p.PRP3fs"}
        );
    }
}
