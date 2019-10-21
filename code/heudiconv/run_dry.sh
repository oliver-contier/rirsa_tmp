#!/usr/bin/env bash

# define base directory of dicom my_data
raw_data_dir="/home/my_data/exppsy/oliver/RIRSA/raw/my_data"

# infer subjects and sessions to iterate over
for subject in $(find $raw_data_dir/session/1/subject -type d -maxdepth 1 -mindepth 1 | xargs -I {} basename {}); do
    for session in $(find $raw_data_dir/session -type d -maxdepth 1 -mindepth 1 | xargs -I {} basename {}); do
        echo $subject
        echo $session

        singularity run \
            /home/data/exppsy/oliver/my_containers/heudiconv/heudiconv-0.5.3.simg \
            -d "/home/my_data/exppsy/oliver/RIRSA/raw/my_data/session/{session}/subject/{subject}/study_*/*" \
            -s ${subject} \
            -ss ${session} \
            -f convertall \
            -c none \
            -o /home/data/exppsy/oliver/RIRSA/scratch/BIDS \
            --overwrite
    done
done
