#!/usr/bin/env bash

# pipe this script to condor_submit to submit every subject as seperate job
# bash submit_subjects.sh | condor_submit

cpu=1                             # CPU cores needed
mem=8000                          # expected memory usage

project_dir=/home/data/exppsy/oliver/RIRSA  # path to project root directory
log_dir=${project_dir}/scratch/logs          # log path
bids_dir=${project_dir}/scratch/BIDS # path to the bids directory
runscript=${project_dir}/raw/code/RSA/glm_group_perm/groupstats_runscript.sh

# create the logs dir if it doesn't exist
[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

# print the .submit header
printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = $cpu
#request_memory = $mem

# Execution
initial_dir    = $project_dir
executable     = $runscript
\n"

# number of contrasts to be parallelized
ncontrasts=3

for contrast in $(seq 1 ${ncontrasts}); do
    for model in 'cat' 'sem' 'V1' 'V2' 'V4' 'IT'; do  # iterate over models
        printf "arguments = ${model} ${contrast}\n"
        printf "log       = ${log_dir}/\$(Cluster).\$(Process).${model}${contrast}.log\n"
        printf "output    = ${log_dir}/\$(Cluster).\$(Process).${model}${contrast}.out\n"
        printf "error     = ${log_dir}/\$(Cluster).\$(Process).${model}${contrast}.err\n"
        printf "Queue\n\n"
    done
done