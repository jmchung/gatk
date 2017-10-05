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
}
