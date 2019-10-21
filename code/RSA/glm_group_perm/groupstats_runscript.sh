#!/usr/bin/env bash

# source FSL
source /etc/fsl/fsl.sh
# source pyhton environment
source /home/data/exppsy/oliver/RIRSA/raw/venvs/rsa_pymvpa_env/bin/activate
# execute python script
python /home/data/exppsy/oliver/RIRSA/raw/code/RSA/glm_group_perm/groupglm_contrastcoded.py $1 $2
