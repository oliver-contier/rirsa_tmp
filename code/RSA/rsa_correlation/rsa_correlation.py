#!/usr/bin/env python

import os
from os.path import join as pjoin

import nltk
import numpy as np
from ..construct_rdms import construct_cornet_rdms, construct_categorical_rdm, construct_semantic_rdm
from ..data_source import get_training_stims, get_events_all, get_conf_regs, get_bold_with_mask, \
    seperate_betas_by_training, split_slcorrr_results
from mvpa2.base.learner import ChainLearner
from mvpa2.datasets.eventrelated import fit_event_hrf_model
from mvpa2.datasets.mri import map2nifti
from mvpa2.mappers.shape import TransposeMapper
from mvpa2.measures.rsa import PDistTargetSimilarity
from mvpa2.measures.searchlight import sphere_searchlight


def main(sub='01',
         ses='1',
         run='1',
         working_directory='/home/data/exppsy/oliver/RIRSA/scratch/rsa/rsa_correlation/workdir',
         bidsdir='/home/data/exppsy/oliver/RIRSA/scratch/BIDS',
         behav_data_dir="/home/data/exppsy/oliver/RIRSA/raw/data/behav_data",
         fmri_logs_dir='/home/data/exppsy/oliver/RIRSA/raw/data/fmri_logs',
         cornet_out_dir="/home/data/exppsy/oliver/RIRSA/raw/data/cornet_output",
         wordnet_download_dir='/home/contier/nltk_data/corpora',
         fmriprep_out_dir='/home/data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out',
         sort_stim_lists=True,
         cornet_layers=('V1', 'V2', 'V4', 'IT'),
         cornet_model_type='S',
         cornet_pdist_metric='correlation',
         cornet_zscore_rdms=True,
         cat_triu=True,
         cat_zscore=True,
         sem_triu=True,
         sem_z_score=True,
         confound_names=('a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
                         'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'),
         hrfmodel='canonical',
         driftmodel='blank',
         glmfit_model='ols',
         rsa_corr_pairwise_metric='correlation',
         rsa_corr_comparison_metric='spearman',
         rsa_corr_coeffonly=False,
         rsa_corr_center=True,
         sl_radius_vox=3,
         n_procs_sl=2):
    """
    Run whole-brain searchlight RSA with pearson correlation between target RDMs and neural RDM
    for a given session of a given subject.

    Using 2 cpus, this is a matter of minutes.
    """

    # make sub-/ses-speciic working directory
    working_sub_directory = pjoin(working_directory, sub, ses, run)
    if not os.path.exists(working_sub_directory):
        os.makedirs(working_sub_directory)

    # Get list of trained vs. untrained stimuli
    trained_obs, untrained_obs = get_training_stims(sub, ses, sort_lists=sort_stim_lists,
                                                    behav_data_dir=behav_data_dir,
                                                    fmri_logs_dir=fmri_logs_dir)
    # get events
    events = get_events_all(sub, ses, run, bids_dir=bidsdir)  # TODO: get also button press regressor

    # get confound regressors
    conf_names, conf_regs = get_conf_regs(sub, ses, run, conf_names=confound_names,
                                          fmriprep_outdir=fmriprep_out_dir)

    # get preprocessed bold data as pymvpa dataset
    bold_ds = get_bold_with_mask(sub, ses, run, fmriprep_outdir=fmriprep_out_dir)

    # fit glm
    print('fitting glm')
    betas = fit_event_hrf_model(bold_ds, events, time_attr='time_coords', condition_attr='condition',
                                design_kwargs=dict(drift_model=driftmodel,
                                                   hrf_model=hrfmodel,
                                                   add_regs=conf_regs,
                                                   add_reg_names=conf_names),
                                glmfit_kwargs=dict(model=glmfit_model),
                                return_model=True)

    # tease apart beta maps for trained and untrained stimuli
    trained_betas, untrained_betas = seperate_betas_by_training(betas, trained_obs, untrained_obs)

    print('getting RDMs')

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

    # Iterate over model RDMs...

    # trained models
    # (categorical is valid for both trained and untrained)
    for model_rdm, model_str in zip(
            [categorical_rdm, trained_sem_rdm] + [trained_cornet_rdms[:, i]
                                                  for i in range(np.shape(trained_cornet_rdms)[1])],
            ['cat', 'sem', 'V1', 'V2', 'V4', 'IT']):

        # set up correlation measure between neural and target rdm
        target_corr = PDistTargetSimilarity(model_rdm, pairwise_metric=rsa_corr_pairwise_metric,
                                            comparison_metric=rsa_corr_comparison_metric,
                                            center_data=rsa_corr_center,
                                            corrcoef_only=rsa_corr_coeffonly)

        # set up searchlight
        sl = sphere_searchlight(ChainLearner([target_corr, TransposeMapper()]), radius=sl_radius_vox,
                                nproc=n_procs_sl)

        print('running trained {} searchlight'.format(model_str))

        # run search light
        sl_result = sl(trained_betas)

        # split correlation coefficient (rho) and p value maps into seperate data sets
        rho_ds, p_ds = split_slcorrr_results(sl_result, bold_ds)

        # save both rho and p to nifti file
        for ds, ds_str in zip([rho_ds, p_ds], ['rho', 'p']):
            mapped = map2nifti(ds)
            mapped.to_filename(
                pjoin(working_sub_directory, '{}_trained_{}_run-{}.nii.gz'.format(ds_str, model_str, run)))

    # untrained models
    for model_rdm, model_str in zip(
            [categorical_rdm, untrained_sem_rdm] + [untrained_cornet_rdms[:, i]
                                                    for i in range(np.shape(untrained_cornet_rdms)[1])],
            ['cat', 'sem', 'V1', 'V2', 'V4', 'IT']):
        # set up correlation measure between neural and target rdm
        target_corr = PDistTargetSimilarity(model_rdm, pairwise_metric=rsa_corr_pairwise_metric,
                                            comparison_metric=rsa_corr_comparison_metric,
                                            center_data=rsa_corr_center,
                                            corrcoef_only=rsa_corr_coeffonly)

        # set up searchlight
        sl = sphere_searchlight(ChainLearner([target_corr, TransposeMapper()]), radius=sl_radius_vox,
                                nproc=n_procs_sl)

        print('running untrained {} searchlight'.format(model_str))

        # run search light
        sl_result = sl(untrained_betas)

        # split correlation coefficient (rho) and p value maps into seperate data sets
        rho_ds, p_ds = split_slcorrr_results(sl_result, bold_ds)

        # save both rho and p to nifti file
        for ds, ds_str in zip([rho_ds, p_ds], ['rho', 'p']):
            mapped = map2nifti(ds)
            mapped.to_filename(
                pjoin(working_sub_directory, '{}_untrained_{}_run-{}.nii.gz'.format(ds_str, model_str, run)))

    return None


if __name__ == '__main__':

    import sys
    subject = sys.argv[1]
    session = sys.argv[2]
    run_ = sys.argv[3]

    main(sub=subject, ses=session, run=run_)
