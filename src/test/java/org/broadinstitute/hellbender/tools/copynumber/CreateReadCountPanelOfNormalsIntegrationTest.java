package org.broadinstitute.hellbender.tools.copynumber;

import org.broadinstitute.hellbender.CommandLineProgramTest;

/**
 * Integration test for {@link CreateReadCountPanelOfNormals}.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class CreateReadCountPanelOfNormalsIntegrationTest extends CommandLineProgramTest {
//    @Test
//    public void testWES() {
////        final File outputPanelOfNormalsFile = createTempFile("create-read-count-panel-of-normals", ".pon");
//        final File outputPanelOfNormalsFile = new File("/home/slee/working/ipython/wes-pon-test/wes.no-gc.pon");
//        final String[] arguments = {
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_0.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_1.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_2.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_3.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_4.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_5.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_6.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_7.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_8.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_9.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_10.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_11.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_12.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_13.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_14.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_15.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_16.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_17.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_18.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_19.tsv",
//                "-" + CopyNumberStandardArgument.NUMBER_OF_EIGENSAMPLES_SHORT_NAME, "20",
//                "-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME, outputPanelOfNormalsFile.getAbsolutePath(),
//                "--" + StandardArgumentDefinitions.VERBOSITY_NAME, "INFO"
//        };
//        runCommandLine(arguments);
//    }
//
//    @Test
//    public void testSparkConverterBug() {
////        final File outputPanelOfNormalsFile = createTempFile("create-read-count-panel-of-normals", ".pon");
//        final File outputPanelOfNormalsFile = new File("/home/slee/working/ipython/wes-pon-test/wes.no-gc.pon");
//        final String[] arguments = {
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-1/execution/TCGA-05-4389-10A-01D-1931-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-10/execution/TCGA-06-0157-10A-01D-1491-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-11/execution/TCGA-06-0214-10A-01D-1491-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-12/execution/TCGA-06-0686-10A-01D-1492-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-13/execution/TCGA-06-0744-10A-01D-1492-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-14/execution/TCGA-06-0745-10A-01D-1492-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-16/execution/TCGA-06-5415-10A-01D-1486-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-17/execution/TCGA-14-2554-10A-01D-1494-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-18/execution/TCGA-19-2620-10A-01D-1495-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-19/execution/TCGA-19-2624-10A-01D-1495-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-20/execution/TCGA-19-2629-10A-01D-1495-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-21/execution/TCGA-26-5132-10A-01D-1486-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-22/execution/TCGA-26-5135-10A-01D-1486-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-23/execution/TCGA-27-2523-10A-01D-1494-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-24/execution/TCGA-27-2528-10A-01D-1494-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-25/execution/TCGA-32-1970-10A-01D-1494-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-26/execution/TCGA-34-5240-10A-01D-1441-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-27/execution/TCGA-41-5651-10A-01D-1696-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-28/execution/TCGA-44-2656-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-29/execution/TCGA-44-2666-10A-01D-0969-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-3/execution/TCGA-05-4395-10A-01D-2364-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-30/execution/TCGA-44-2666-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-31/execution/TCGA-44-2668-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-32/execution/TCGA-44-3917-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-33/execution/TCGA-44-3918-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-34/execution/TCGA-44-4112-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-35/execution/TCGA-44-5645-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-36/execution/TCGA-44-6146-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-37/execution/TCGA-44-6147-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-38/execution/TCGA-44-6775-10A-01D-A46W-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-39/execution/TCGA-50-5066-10A-01D-1625-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-40/execution/TCGA-55-7281-10A-01D-2036-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-41/execution/TCGA-64-1678-10A-01D-1040-01.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-42/execution/TCGA-64-1680-10A-01D-1040-01.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-43/execution/TCGA-67-6215-10A-01D-1753-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-44/execution/TCGA-75-6203-10A-01D-1753-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-5/execution/TCGA-05-4396-10A-01D-1855-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-7/execution/TCGA-05-4420-10A-01D-1931-08.coverage.tsv.raw_cov.hdf5",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/dsde/working/slee/wgs-coverage-250-1000-4000/cromwell-executions/CNVCoverageWorkflow/eb80393a-6e31-47bb-a5bf-8ad945157afa/call-Coverage3000NoDupes/shard-9/execution/TCGA-05-4422-10A-01D-1931-08.coverage.tsv.raw_cov.hdf5",
//                "-" + CreateReadCountPanelOfNormals.MINIMUM_INTERVAL_MEDIAN_PERCENTILE_SHORT_NAME, "0",
//                "-" + CreateReadCountPanelOfNormals.MAXIMUM_ZEROS_IN_SAMPLE_PERCENTAGE_SHORT_NAME, "100",
//                "-" + CreateReadCountPanelOfNormals.MAXIMUM_ZEROS_IN_INTERVAL_PERCENTAGE_SHORT_NAME, "100",
//                "-" + CreateReadCountPanelOfNormals.EXTREME_SAMPLE_MEDIAN_PERCENTILE_SHORT_NAME, "0",
//                "-" + CreateReadCountPanelOfNormals.EXTREME_OUTLIER_TRUNCATION_PERCENTILE_SHORT_NAME, "0",
//                "-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME, outputPanelOfNormalsFile.getAbsolutePath(),
//                "--" + StandardArgumentDefinitions.VERBOSITY_NAME, "INFO"
//        };
//        runCommandLine(arguments);
//    }
//
//    @Test
//    public void testWESWithGC() {
////        final File outputPanelOfNormalsFile = createTempFile("create-read-count-panel-of-normals", ".pon");
//        final File outputPanelOfNormalsFile = new File("/home/slee/working/ipython/wes-pon-test/wes.gc.pon");
//        final String[] arguments = {
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_0.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_1.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_2.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_3.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_4.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_5.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_6.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_7.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_8.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_9.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_10.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_11.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_12.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_13.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_14.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_15.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_16.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_17.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_18.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes_19.tsv",
//                "-" + TargetArgumentCollection.TARGET_FILE_SHORT_NAME, "/home/slee/working/ipython/wes-pon-test/wes.intervals.annot.tsv",
//                "-" + CopyNumberStandardArgument.NUMBER_OF_EIGENSAMPLES_SHORT_NAME, "20",
//                "-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME, outputPanelOfNormalsFile.getAbsolutePath(),
//                "--" + StandardArgumentDefinitions.VERBOSITY_NAME, "INFO"
//        };
//        runCommandLine(arguments);
//    }
//
//    @Test
//    public void testWGS5M() {
////        final File outputPanelOfNormalsFile = createTempFile("create-read-count-panel-of-normals", ".pon");
//        final File outputPanelOfNormalsFile = new File("/home/slee/working/ipython/wgs-pon-test-5M/wgs-5M.no-gc.pon");
//        final String[] arguments = {
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test-5M/wgs_0.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test-5M/wgs_1.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test-5M/wgs_2.tsv",
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test-5M/wgs_3.tsv",
//                "-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME, outputPanelOfNormalsFile.getAbsolutePath(),
//                "--" + StandardArgumentDefinitions.VERBOSITY_NAME, "INFO"
//        };
//        runCommandLine(arguments);
//    }
//
//    @Test
//    public void testWGS() {
////        final File outputPanelOfNormalsFile = createTempFile("create-read-count-panel-of-normals", ".pon");
//        final File outputPanelOfNormalsFile = new File("/home/slee/working/ipython/wgs-pon-test/wgs.no-gc.pon");
//        final String[] arguments = {
//                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_0.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_1.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_2.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_3.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_4.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_5.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_6.tsv",
////                "-" + StandardArgumentDefinitions.INPUT_SHORT_NAME, "/home/slee/working/ipython/wgs-pon-test/wgs_7.tsv",
//                "-" + CreateReadCountPanelOfNormals.MAXIMUM_ZEROS_IN_SAMPLE_PERCENTAGE_SHORT_NAME, "10",
//                "-" + CreateReadCountPanelOfNormals.MAXIMUM_ZEROS_IN_INTERVAL_PERCENTAGE_SHORT_NAME, "50",
//                "-" + StandardArgumentDefinitions.OUTPUT_SHORT_NAME, outputPanelOfNormalsFile.getAbsolutePath(),
//                "--" + StandardArgumentDefinitions.VERBOSITY_NAME, "INFO"
//        };
//        runCommandLine(arguments);
//    }
}