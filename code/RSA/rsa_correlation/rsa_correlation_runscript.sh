#!/usr/bin/env bash

# source pyhton environment
source /home/data/exppsy/oliver/RIRSA/raw/venvs/rsa_pymvpa_env/bin/activate

# run pyhton script for subject and session arguments
#python /home/data/exppsy/oliver/RIRSA/raw/code/RSA/rsa_correlation/rsa_correlation.py $1 $2 $3
cd /home/data/exppsy/oliver/RIRSA/raw/code/
python -m RSA.rsa_correlation.rsa_correlation $1 $2 $3