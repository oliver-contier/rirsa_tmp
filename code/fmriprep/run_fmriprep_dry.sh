#!/usr/bin/env bash

singularity run --cleanenv /home/data/exppsy/oliver/my_containers/fmriprep/fmriprep-1.3.1.simg \
--fs-license-file ~/freesurfer_license/license.txt --reports-only --write-graph \
-w /home/data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_work \
/home/data/exppsy/oliver/RIRSA/scratch/BIDS/ \
/home/data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out \
participant

# --task-id
# --n_cpus
# --mem_mb
# --use-plugin
# --verbose
# --output-space {T1w, template, fsnative, fsaverage, fsaverage6, fsaverage5}
# --longitudinal
# --template {MNI152NLin2009cAsym}  # only one choice?
# --use-aroma  --aroma-melodic-dimensionality
# –fs-no-reconall  # omit freesurfer surface reconstruction
# --resource-monitor
# --reports-only  # run dry
# --write-graph
