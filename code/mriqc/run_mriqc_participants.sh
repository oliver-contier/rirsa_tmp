#!/usr/bin/env bash

singularity run \
    /home/data/exppsy/oliver/my_containers/mriqc/mriqc-0.14.2 \
    /home/data/exppsy/oliver/RIRSA/scratch/BIDS \
    /home/data/exppsy/oliver/RIRSA/scratch/mriqc/mriqc_out \
    participant \
    --verbose \
    --verbose-reports \
    --use-plugin CondorDAGMan \
    --work-dir /home/data/exppsy/oliver/RIRSA/scratch/mriqc/mriqc_workdir

    #--participant_label 01