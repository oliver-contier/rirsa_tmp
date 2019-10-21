#!/bin/sh

# pipe this script to condor_submit to submit every subject as seperate job
# bash submit_subjects.sh | condor_submit

cpu=4                             # CPU cores needed
mem=8000                          # expected memory usage

project_dir=/home/data/exppsy/oliver/RIRSA  # path to project root directory
log_dir=${project_dir}/scratch/logs          # log path
bids_dir=${project_dir}/scratch/BIDS # path to the bids directory
runscript=${project_dir}/raw/code/RSA/pre_glm/pre_glm_runscript.sh

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

for session in 1 2; do  # iterate over sessions
    for dir in ${bids_dir}/sub-*/ ; do  # iterate over subjects (inferred from bids directory)
        subject=$(basename ${dir//sub-})
        printf "arguments = ${subject} ${session}\n"
        printf "log       = ${log_dir}/\$(Cluster).\$(Process).${subject}.log\n"
        printf "output    = ${log_dir}/\$(Cluster).\$(Process).${subject}.out\n"
        printf "error     = ${log_dir}/\$(Cluster).\$(Process).${subject}.err\n"
        printf "Queue\n\n"
    done
done