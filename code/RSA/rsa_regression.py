#!/usr/bin/env python
"""
RSA procedure for an individual subject.

General set-up:

+ 1. Get list of trained vs. untrained stim
+ 2. Construct model RDM's
    + a) CorNet
    + b) categorical
    + c) semantic (translated to english)

+ 1. load stimulus events
+ load confound events
+ 2. load brain my_data (with mask)
+ 3. fit hrf dataset (with events + nuisance regs)
4. set up within-searchlight-processes
    distance measure for neural RDM (ideally, CV-Mahalanobis), z-transformed for normal distribution
    model-fitting procedure (partial, semipartial, or regression, rank / Kandell's Tau)
5. set up searchlight itself
6. run sl

In other scripts:
- RSA post-processing (smoothing)
- within-sub GLM
- group GLM
"""

import os
import time
from os.path import join as pjoin

import nltk
from mvpa2.base.hdf5 import h5load
from mvpa2.clfs.stats import MCNullDist
from mvpa2.datasets.eventrelated import fit_event_hrf_model
from mvpa2.datasets.mri import map2nifti
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.measures.rsa import Regression
from mvpa2.measures.searchlight import sphere_searchlight

from .construct_rdms import construct_cornet_rdms, construct_categorical_rdm, construct_semantic_rdm, \
    makemodel_IT_semantic_categorical
from .data_source import get_training_stims, get_events_all, get_conf_regs, get_bold_with_mask, \
    seperate_betas_by_training, split_sl_results_it_sem_cat


def main(sub='01',
         ses='1',
         run='1',
         working_directory='/home/my_data/exppsy/oliver/RIRSA/scratch/rsa/rsa_regression/workdir',
         load_from_workdir=False,
         bidsdir='/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS',
         behav_data_dir="/home/my_data/exppsy/oliver/RIRSA/raw/my_data/behav_data",
         fmri_logs_dir='/home/my_data/exppsy/oliver/RIRSA/raw/my_data/fmri_logs',
         cornet_out_dir="/home/my_data/exppsy/oliver/RIRSA/raw/my_data/cornet_output",
         wordnet_download_dir='/home/contier/nltk_data/corpora',
         fmriprep_out_dir='/home/my_data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out',
         sort_stim_lists=True,
         cornet_layers=('V1', 'V2', 'V4', 'IT'),
         cornet_model_type='S',
         cornet_pdist_metric='correlation',
         cornet_zscore_rdms=True,
         cat_triu=True,
         cat_zscore=False,
         sem_triu=True,
         sem_z_score=True,
         confound_names=('a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
                         'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'),
         hrfmodel='canonical',
         driftmodel='blank',
         glmfit_model='ols',
         rsa_reg_metric='correlation',
         rsa_reg_npermutations=5,
         rsa_reg_method='ridge',
         rsa_reg_center=True,
         rsa_reg_rank=True,
         rsa_reg_normalize=True,
         rsa_reg_fitint=True,
         sl_radius=4,
         n_procs_sl=2):
    """
    Run RSA analysis for a given run, session, and subject.
    """

    """
    Load my_data
    """
    # make sub-/ses-speciic working directory
    working_sub_directory = pjoin(working_directory, sub, ses)
    if not os.path.exists(working_sub_directory):
        os.makedirs(working_sub_directory)
    print('loading my_data')
    # Get list of trained vs. untrained stimuli
    trained_obs, untrained_obs = get_training_stims(sub, ses, sort_lists=sort_stim_lists,
                                                    behav_data_dir=behav_data_dir,
                                                    fmri_logs_dir=fmri_logs_dir)
    # get events
    # TODO: get also button press regressor
    events = get_events_all(sub, ses, run, bids_dir=bidsdir)
    # get confound regressors
    conf_names, conf_regs = get_conf_regs(sub, ses, run, conf_names=confound_names,
                                          fmriprep_outdir=fmriprep_out_dir)
    # get preprocessed bold my_data as pymvpa ds
    bold_ds = get_bold_with_mask(sub, ses, run, fmriprep_outdir=fmriprep_out_dir)

    """
    RDMs
    """
    print('creating RDMs')
    # Get cornet RDMs
    trained_cornet_rdms = construct_cornet_rdms(trained_obs,
                                                cornet_out_dir=cornet_out_dir,
                                                layers=cornet_layers,
                                                model_type=cornet_model_type,
                                                pdist_metric=cornet_pdist_metric,
                                                zscore_rdms=cornet_zscore_rdms)
    untrained_cornet_rdms = construct_cornet_rdms(untrained_obs,
                                                  cornet_out_dir=cornet_out_dir,
                                                  layers=cornet_layers,
                                                  model_type=cornet_model_type,
                                                  pdist_metric=cornet_pdist_metric,
                                                  zscore_rdms=cornet_zscore_rdms)
    # Get categorical RDM
    categorical_rdm = construct_categorical_rdm(trained_obs, z_score=cat_zscore, return_triu=cat_triu)
    # download wordnet corpus if necessary
    if wordnet_download_dir:
        nltk.download('wordnet', download_dir=wordnet_download_dir)
    # Get semantic RDMs
    trained_sem_rdm = construct_semantic_rdm(trained_obs, z_score=sem_z_score, return_triu=sem_triu)
    untrained_sem_rdm = construct_semantic_rdm(untrained_obs, z_score=sem_z_score, return_triu=sem_triu)

    """
    Fit GLM
    """
    print('fitting glm')
    # fit glm
    betas = fit_event_hrf_model(bold_ds, events, time_attr='time_coords', condition_attr='condition',
                                design_kwargs=dict(drift_model=driftmodel,
                                                   hrf_model=hrfmodel,
                                                   add_regs=conf_regs,
                                                   add_reg_names=conf_names),
                                glmfit_kwargs=dict(model=glmfit_model),
                                return_model=True)
    # tease apart trained from untrained betas
    print('seperating betas')
    trained_betas, untrained_betas = seperate_betas_by_training(betas, trained_obs, untrained_obs)

    """
    set up RSA regression
    """
    print('setting up searchlight rsa')
    # general: set up permutation, null distribution
    permutator = AttributePermutator('condition', count=rsa_reg_npermutations)
    mcnull = MCNullDist(permutator, tail='right', enable_ca=['ca.dist_samples'])
    # TODO: note that this tests only right tailed
    #  (though I can't get even that to work)

    # set up RSA regression models and settings
    trained_model = makemodel_IT_semantic_categorical(trained_cornet_rdms, trained_sem_rdm, categorical_rdm)
    trained_rsa_reg = Regression(trained_model, pairwise_metric=rsa_reg_metric, center_data=rsa_reg_center,
                                 method=rsa_reg_method, fit_intercept=rsa_reg_fitint, rank_data=rsa_reg_rank,
                                 normalize=rsa_reg_normalize, null_dist=mcnull, keep_pairs=None)
    untrained_model = makemodel_IT_semantic_categorical(untrained_cornet_rdms, untrained_sem_rdm, categorical_rdm)
    untrained_rsa_reg = Regression(untrained_model, pairwise_metric=rsa_reg_metric, center_data=rsa_reg_center,
                                   method=rsa_reg_method, fit_intercept=rsa_reg_fitint, rank_data=rsa_reg_rank,
                                   normalize=rsa_reg_normalize, null_dist=mcnull, keep_pairs=None)

    """
    Run searchlight(s)
    """
    if load_from_workdir:
        trained_results = h5load(pjoin(working_sub_directory, 'trained_results.hd5'))
        untrained_results = h5load(pjoin(working_sub_directory, 'untrained_results.hd5'))
    else:
        # trained
        time_start = time.time()
        print('started running first searchlight at {}'.format(time_start))
        trained_sl = sphere_searchlight(trained_rsa_reg, radius=sl_radius, nproc=n_procs_sl)
        print('running searchlight for trained stimuli')
        trained_results = trained_sl(trained_betas)
        print('finished running first earchlight after {}'.format(time.time() - time_start))
        print('saving trained_results')
        trained_results.save(pjoin(working_sub_directory, 'trained_results.hd5'))

        # untrained
        untrained_sl = sphere_searchlight(untrained_rsa_reg, radius=sl_radius, nproc=n_procs_sl)
        print('running searchlight for trained stimuli')
        untrained_results = untrained_sl(untrained_betas)
        print('saving trained_results')
        untrained_results.save(pjoin(working_sub_directory, 'untrained_results.hd5'))

    """
    Reshape, convert and save results
    """
    trained_it, trained_sem, trained_cat = split_sl_results_it_sem_cat(trained_results, mapper_from=bold_ds)
    untrained_it, untrained_sem, untrained_cat = split_sl_results_it_sem_cat(untrained_results, mapper_from=bold_ds)

    for results_map, fname in zip([trained_it, trained_sem, trained_cat,
                                   untrained_it, untrained_sem, untrained_cat],
                                  ['trained_it', 'trained_sem', 'trained_cat',
                                   'untrained_it', 'untrained_sem', 'untrained_cat']):
        mapped = map2nifti(results_map)
        mapped.to_filename(pjoin(working_sub_directory, fname + '.nii.gz'))

    return None


if __name__ == '__main__':
    main(load_from_workdir=True)
