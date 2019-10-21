#!/usr/bin/env bash

# argument for --participant_label is passed in from condor submission file

singularity run \
    --cleanenv \
    --bind /home/data/exppsy/oliver/RIRSA/:/proj_root \
    --bind /home/contier/freesurfer_license:/fslicense \
    /home/data/exppsy/oliver/my_containers/fmriprep/fmriprep-1.3.1.simg \
        /proj_root/scratch/BIDS \
        /proj_root/scratch/fmriprep/fmriprep_out \
        participant \
            -w /proj_root/scratch/fmriprep/fmriprep_work \
            --fs-license-file /fslicense/license.txt \
            --fs-no-reconall \
            --output-space T1w template \
            --participant_label $1


# singularity run --cleanenv  # exclude environment paths of cluster
# --task-id
# --participant_label
# --n_cpus
# --mem_mb
# --use-plugin {nipype.cfg file}
# --verbose
# --output-space {T1w, template, fsnative, fsaverage, fsaverage6, fsaverage5}
# --longitudinal
# --template {MNI152NLin2009cAsym}  # only one choice?
# --use-aroma  --aroma-melodic-dimensionality
# –fs-no-reconall  # omit freesurfer surface reconstruction
# --resource-monitor
# --reports-only  # run dry
# --write-graph