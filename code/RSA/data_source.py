#!/usr/bin/env python

import copy
import csv
from os.path import join as pjoin

import numpy as np
import pandas as pd
from mvpa2.tutorial_suite import fmri_dataset


def get_training_stims(sub_id,
                       sess_id,
                       behav_data_dir='/home/my_data/exppsy/oliver/RIRSA/raw/my_data/behav_data',
                       fmri_logs_dir='/home/my_data/exppsy/oliver/RIRSA/raw/my_data/fmri_logs',
                       sort_lists=True):
    """
    Get the set of trained and untrained stimuli objects from the behavioral results csv file
    for a given subject.
    """
    # get trained and untrained object IDs
    behav_fname = pjoin(behav_data_dir, 'sub{}_behav.csv'.format(sub_id))
    behav_df = pd.read_csv(behav_fname)
    fmrilog_fname = pjoin(fmri_logs_dir, 'sub{}_session{}_rionly_run1_fmri.csv'.format(sub_id, sess_id))
    fmrilog_df = pd.read_csv(fmrilog_fname)
    trained_obs = [obj for obj in behav_df.object_id.unique()]
    all_obs = fmrilog_df.object_id.unique()
    untrained_obs = [obj for obj in all_obs if obj not in trained_obs]
    # add rotations
    rotations = range(0, 316, 45)  # from 0 to 315 in 45 degree steps
    trained_obs = np.array([str(obj) + '_r{}'.format(str(rot)) for rot in rotations for obj in trained_obs])
    untrained_obs = np.array([str(obj) + '_r{}'.format(str(rot)) for rot in rotations for obj in untrained_obs])
    assert len(trained_obs) == len(untrained_obs) and len(trained_obs) == 8 * 8
    if sort_lists:
        trained_obs.sort()
        untrained_obs.sort()
    return trained_obs, untrained_obs


def get_events_all(sub,
                   ses,
                   run,

                   bids_dir='/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS'):
    """
    Get all stimulus events for a given subject, session, and run.
    """
    events_tsv = pjoin(bids_dir, 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
                       'sub-{}_ses-{}_task-ri_run-{}_events.tsv'.format(sub, ses, run))
    with open(events_tsv) as f:
        csvreader = csv.reader(f, delimiter='\t')
        tsv_content = [row for row in csvreader]
    events = [
        {'onset': float(row[0]), 'duration': float(row[1]), 'condition': row[2]}
        for row in tsv_content[1:]]
    return events


def get_events_for_object_ids(sub,
                              ses,
                              run,
                              object_ids,
                              bids_dir='/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS'):
    """
    Get a pymvpa-compatible dict with events for a set ob object_ids
    given a subject, session, and run.
    """
    events_tsv = pjoin(bids_dir, 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
                       'sub-{}_ses-{}_task-ri_run-{}_events.tsv'.format(sub, ses, run))
    with open(events_tsv) as f:
        csvreader = csv.reader(f, delimiter='\t')
        tsv_content = [row for row in csvreader]
    events = [
        {'onset': float(row[0]), 'duration': float(row[1]), 'condition': row[2]}
        for row in tsv_content[1:] if row[2] in object_ids]
    return events


def get_conf_regs(sub,
                  ses,
                  run,
                  conf_names=('a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
                              'a_comp_cor_03', 'a_comp_cor_04', 'trans_x', 'trans_y',
                              'trans_z', 'rot_x', 'rot_y', 'rot_z'),
                  fmriprep_outdir='/home/my_data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out'):
    """
    Grab confound regressors from fmriprep output.
    """
    confound_tsv = pjoin(fmriprep_outdir, 'fmriprep', 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
                         'sub-{}_ses-{}_task-ri_run-{}_desc-confounds_regressors.tsv'.format(sub, ses, run))
    with open(confound_tsv) as f:
        csvreader = csv.reader(f, delimiter='\t')
        tsv_content = [row for row in csvreader]
    reg_idxs = [tsv_content[0].index(confname) for confname in conf_names]
    conf_regs = np.array(tsv_content[1:])[:, reg_idxs].astype('float')
    assert np.shape(conf_regs) == (175, len(conf_names))
    return list(conf_names), conf_regs


def get_bold_with_mask(sub,
                       ses,
                       run,
                       fmriprep_outdir='/home/my_data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out'):
    """
    load preprocessed bold image together with mask image (both returned from fmriprep) into pymvpa dataset
    """
    mask_path = pjoin(fmriprep_outdir, 'fmriprep', 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
                      'sub-{}_ses-{}_task-ri_run-{}_space-T1w_desc-brain_mask.nii.gz'.format(sub, ses, run))
    bold_path = pjoin(fmriprep_outdir, 'fmriprep', 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
                      'sub-{}_ses-{}_task-ri_run-{}_space-T1w_desc-preproc_bold.nii.gz'.format(sub, ses, run))
    fds = fmri_dataset(bold_path, mask=mask_path)
    return fds


def _find_beta_indices(beta_ds,
                       trained_objects,
                       untrained_objects):
    """
    find indices for trained / untrained stimuli in beta_ds.
    used for seperating beta maps in function seperate_betas_by_training
    """
    trained_indices, untrained_indices = [], []
    for i in range(len(beta_ds.sa.condition)):
        if beta_ds.sa.condition[i] == 'catch':
            continue
        elif beta_ds.sa.condition[i] in trained_objects:
            trained_indices.append(i)
        elif beta_ds.sa.condition[i] in untrained_objects:
            untrained_indices.append(i)
        else:
            raise RuntimeError('couldnt figure out object {}'.format(beta_ds.sa.condition[i]))
    assert len(trained_indices) == len(trained_objects) and len(untrained_indices) == len(untrained_objects)
    return np.array(trained_indices), np.array(untrained_indices)


def seperate_betas_by_training(beta_ds,
                               trained_objects,
                               untrained_objects):
    """
    Given a dataset of beta values obtained via fit_event_hrf_model,
    create two new ones for trained and untrained stimulus responses respectively.
    """
    # find indices of trained vs. untrained_stimuli
    trained_indices, untrained_indices = _find_beta_indices(beta_ds, trained_objects, untrained_objects)
    # make copies of beta_ds without regressors and condition (because they throw errors due to shape)
    # but keep all the rest
    keep_sa = []
    keep_fa = ['voxel_indices']
    keep_a = ['mapper', 'imgaffine', 'imghdr', 'voxel_dim', 'imgtype', 'voxel_eldim']
    trained_betas = beta_ds.copy(deep=False, sa=keep_sa, fa=keep_fa, a=keep_a)
    untrained_betas = beta_ds.copy(deep=False, sa=keep_sa, fa=keep_fa, a=keep_a)
    # add selected samples (i.e. beta values), regressor and condition info
    for indices, betamap in zip([trained_indices, untrained_indices],
                                [trained_betas, untrained_betas]):
        betamap.samples = beta_ds.samples[indices, :]
        betamap.sa.regressors = beta_ds.sa.regressors[indices, :]
        betamap.sa.condition = beta_ds.sa.condition[indices]
    return trained_betas, untrained_betas


def split_sl_results_it_sem_cat(sl_results):
    # initiate empty my_data sets
    # TODO: used in rsa regression and probably not working in this state
    it_results, sem_results, cat_results = copy.deepcopy(sl_results), \
                                           copy.deepcopy(sl_results), \
                                           copy.deepcopy(sl_results)
    # add correct parameter estimates to them plus some attributes
    for idx, resultsmap in enumerate([it_results, sem_results, cat_results]):
        resultsmap.samples = sl_results.samples[idx, :]
        # resultsmap.sa['coefs'] = sl_results.sa.coefs[idx]
        # resultsmap.fa['center_ids'] = sl_results.fa.center_ids
        # resultsmap.a['mapper'] = mapper_from.a.mapper
    return it_results, sem_results, cat_results


def split_slcorrr_results(slresults, boldds):
    """
    Take the results from running a RSA-correlation searchlight analysis,
    split rho and p maps into seperate data sets,
    and save it with image information taken from the original bold dataset on which the analysis was run.
    """
    # make copies
    rho_ds = slresults.copy(deep=False)
    p_ds = slresults.copy(deep=False)
    # take only subset of samples (for selecting rho and p values)
    for idx, target_ds in enumerate([rho_ds, p_ds]):
        target_ds.samples = slresults.samples[idx, :]
        # add general image attributes from bold dataset (for saving in correct space)
        for copy_attr in ['mapper', 'imgaffine', 'imghdr', 'voxel_dim', 'imgtype', 'voxel_eldim']:
            target_ds.a[copy_attr] = boldds.a[copy_attr]
    return rho_ds, p_ds
