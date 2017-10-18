## Running the Somatic CNV WDL

### Which WDL should you use?
- Building a panel of normals (PoN): ``cnv_somatic_panel_workflow.wdl``
- Running a matched pair: ``cnv_somatic_pair_workflow.wdl``

#### Setting up parameter json file for a run

To get started, copy the relevant ``*.template.json`` for the workflow you wish to run and adjust parameters accordingly.  
- Values starting with ``$__`` *must* be replaced with values for your run.  Please note that python [templates](https://docs.python.org/2/library/string.html#template-strings) can be useful for replacing these values.

*Please note that there are task-level parameters that do not appear in the template files.  These are set to reasonable values by default, but can also be adjusted if desired.*

#### Explanation of fields in the somatic panel workflow

The reference used must be the same between PoN and case samples.

- ``CNVSomaticPanelWorkflow.targets`` -- (optional) Target file (NOT in bed format) that was used to describe the baits in capture (exome) samples.  Please run ``ConvertBedToTargetFile`` to convert a BED file to a target file.  If provided, then WES workflow will be run; otherwise, WGS workflow will be run.
- ``CNVSomaticPanelWorkflow.normal_bams_list`` -- TSV file consisting of corresponding bam and corresponding index files as described in cnv_somatic_panel_workflow.wdl.
- ``CNVSomaticPanelWorkflow.pon_entity_id`` -- Name of the final PoN file.
- ``CNVSomaticPanelWorkflow.ref_fasta`` -- Path to reference fasta file.
- ``CNVSomaticPanelWorkflow.ref_fasta_fai`` -- Path to reference fasta fai file.
- ``CNVSomaticPanelWorkflow.ref_fasta_dict`` -- Path to reference dict file.
- ``CNVSomaticPanelWorkflow.do_explicit_gc_correction`` -- (optional) If true, perform explicit GC-bias correction when creating PoN and in subsequent denoising of case samples.  If false, rely on PCA-based denoising to correct for GC bias.
- ``CNVSomaticPanelWorkflow.gatk_jar`` -- Absolute path to gatk.jar.
- ``CNVSomaticPanelWorkflow.gatk_docker`` -- GATK Docker image (e.g., ``broadinstitute/gatk:x.beta.x``).

In additional, there are several task-level parameters that may be set by advanced users; for example:

- ``CNVSomaticPanelWorkflow.CollectReadCounts.wgs_bin_length`` -- Size of bins (in bp) for WGS coverage collection.  *This must be the same value used for all case samples.*  Ignored if not running WGS.
- ``CNVSomaticPanelWorkflow.PadTargets.padding`` -- Amount of padding (in bp) to add to both sides of targets for WES coverage collection.  *This must be the same value used for all case samples.*  Ignored if not running WES.

Further explanation of other task-level parameters may be found by invoking the ``--help`` documentation available in the gatk.jar for each tool.  

#### Explanation of fields in the somatic case workflow

The reference (and targets, if specified) used must be the same between PoN and case samples.

- ``CNVSomaticPairWorkflow.targets`` -- (optional) Target file (NOT in bed format) that was used to describe the baits in capture (exome) samples.  Please run ``ConvertBedToTargetFile`` to convert a BED file to a target file.  If provided, then WES workflow will be run; otherwise, WGS workflow will be run.
- ``CNVSomaticPairWorkflow.common_sites`` -- Picard or GATK-style interval list of common sites to use for collecting allelic counts.
- ``CNVSomaticPairWorkflow.read_count_pon`` -- Path to read-count PoN created by the panel workflow. 
- ``CNVSomaticPairWorkflow.tumor_bam`` -- File path or storage location (depending on backend) of the tumor BAM file.
- ``CNVSomaticPairWorkflow.tumor_bam_idx`` -- File path or storage location (depending on backend) of the tumor BAM file index.
- ``CNVSomaticPairWorkflow.normal_bam`` -- File path or storage location (depending on backend) of the normal BAM file.
- ``CNVSomaticPairWorkflow.normal_bam_idx`` -- File path or storage location (depending on backend) of the normal BAM file index.
- ``CNVSomaticPairWorkflow.ref_fasta`` -- Path to reference fasta file.
- ``CNVSomaticPairWorkflow.ref_fasta_fai`` -- Path to reference fasta fai file.
- ``CNVSomaticPairWorkflow.ref_fasta_dict`` -- Path to reference dict file.
- ``CNVSomaticPairWorkflow.gatk_jar`` -- Absolute path to gatk.jar.
- ``CNVSomaticPairWorkflow.gatk_docker`` -- GATK Docker image (e.g., "broadinstitute/gatk:x.beta.x").

In additional, there are several task-level parameters that may be set by advanced users as above.

Further explanation of these task-level parameters may be found by invoking the ``--help`` documentation available in the gatk.jar for each tool.