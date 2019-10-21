#!/usr/bin/env bash

# source FSL
source /etc/fsl/fsl.sh
# source pyhton environment
source /home/data/exppsy/oliver/RIRSA/raw/venvs/rsa_pymvpa_env/bin/activate
# run pyhton script for subject and session arguments
python /home/data/exppsy/oliver/RIRSA/raw/code/RSA/pre_glm/pre_glm.py $1 $2