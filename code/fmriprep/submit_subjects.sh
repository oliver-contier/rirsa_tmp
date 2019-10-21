#!/bin/sh

# pipe this script to condor_submit to submit every subject as seperate job
# bash submit_subjects.sh | condor_submit

cpu=1                             # CPU cores needed
mem=4000                          # expected memory usage

project_dir=/home/data/exppsy/oliver/RIRSA  # path to project root directory
log_dir=${project_dir}/scratch/logs          # log path
bids_dir=${project_dir}/scratch/BIDS # path to the bids directory
analysis_script=${project_dir}/raw/code/fmriprep/run_fmriprep.sh  # path to run script to which subject arg is passed

# create the logs dir if it doesn't exist
[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

# print the .submit header
printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = $cpu
request_memory = $mem

# Execution
initial_dir    = $project_dir
executable     = $analysis_script
\n"

# create a job for each subject file
for dir in ${bids_dir}/sub-*/ ; do
    subject=$(basename ${dir//sub-})
    printf "arguments = ${subject}\n"
    printf "log       = ${log_dir}/\$(Cluster).\$(Process).${subject}.log\n"
    printf "output    = ${log_dir}/\$(Cluster).\$(Process).${subject}.out\n"
    printf "error     = ${log_dir}/\$(Cluster).\$(Process).${subject}.err\n"
    printf "Queue\n\n"
done