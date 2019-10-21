#!/usr/bin/env python

"""
Run a GLM to get stimulus-specific responses (betas) within each run and agregate these results
across all runs in a given session.

This script is executed for a given session of a given subject.
"""

import nipype.pipeline.engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function
from nipype.workflows.fmri.fsl import create_modelfit_workflow, create_fixed_effects_flow


def grab_data(sub,
              ses,
              bidsdir,
              fmriprepoutdir,
              fmrilogsdir,
              nruns=4):
    """
    Get file paths of preprocessed bold data, the brain mask in anatomical space,
    events.tsv files, confound.tsv files, fmrilog.csv files (for all runs in a session of given subject).
    """
    from os.path import join as pjoin
    preproc_bold_runs = [
        pjoin(fmriprepoutdir, 'fmriprep', 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
              'sub-{}_ses-{}_task-ri_run-{}_space-T1w_desc-preproc_bold.nii.gz'.format(sub, ses, run))
        for run in range(1, nruns + 1)]
    event_tsvs = [
        pjoin(bidsdir, 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
              'sub-{}_ses-{}_task-ri_run-{}_events.tsv'.format(sub, ses, run)) for run in range(1, nruns + 1)]
    confound_tsvs = [
        pjoin(fmriprepoutdir, 'fmriprep', 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
              'sub-{}_ses-{}_task-ri_run-{}_desc-confounds_regressors.tsv'.format(sub, ses, run))
        for run in range(1, nruns + 1)]
    fmrilog_csvs = [
        pjoin(fmrilogsdir, 'sub{}_session{}_rionly_run{}_fmri.csv'.format(sub, ses, run))
        for run in range(1, nruns + 1)]
    assert len(event_tsvs) == len(confound_tsvs) and len(event_tsvs) == len(fmrilog_csvs)
    design_tsv_zips = zip(event_tsvs, confound_tsvs, fmrilog_csvs)
    bold_masks_per_run = [pjoin(fmriprepoutdir, 'fmriprep', 'sub-{}'.format(sub), 'ses-{}'.format(ses), 'func',
                                'sub-{}_ses-{}_task-ri_run-{}_space-T1w_desc-brain_mask.nii.gz'.format(sub, ses, run))
                          for run in range(1, nruns + 1)]
    anat_mask = pjoin(fmriprepoutdir, 'fmriprep', 'sub-{}'.format(sub), 'anat',
                      'sub-{}_desc-brain_mask.nii.gz'.format(sub))
    return preproc_bold_runs, design_tsv_zips, bold_masks_per_run, anat_mask


def event_tsvs_to_bunch(design_tsv_zip,
                        sort_events,
                        confound_names,
                        keep_confound_names=True,
                        response_duration_fixed_at=.2):
    """
    Take condition and nuisance regressor information from tsv files and return them in form of a "Bunch" object,
    which can be used by SpecifyModel.
    """
    import pandas as pd
    import numpy as np
    import csv
    from nipype.interfaces.base import Bunch
    # tease apart zipped file paths
    events_tsv, confounds_tsv, fmrilog_csv = design_tsv_zip
    # get events (conditions, onsets, durations)
    events_df = pd.read_csv(events_tsv, sep="\t")
    conditions = events_df['trial_type'].unique().tolist()
    assert len(conditions) == 129
    if sort_events:
        conditions.sort()
    onsets = [list(events_df[events_df['trial_type'] == cond]['onset'].values.tolist())
              for cond in conditions]
    durations = [list(events_df[events_df['trial_type'] == cond]['duration'].values.tolist())
                 for cond in conditions]
    # open fmri_log.csv
    with open(fmrilog_csv) as f:
        csvreader = csv.reader(f, delimiter=',')
        fmrilog_content = [row for row in csvreader]
    # get onsets of motor responses from fmrilog content
    rt_idx = fmrilog_content[0].index('RT')
    global_onset_idx = fmrilog_content[0].index('global_onset_time')
    response_onsets = [float(row[rt_idx]) + float(row[global_onset_idx])
                       for row in fmrilog_content[1:]
                       if row[rt_idx] != '']
    # append to other conditions, but only if at least one response was given (we don't want flat regressors)
    if response_onsets:
        onsets.append(response_onsets)
        durations.append([response_duration_fixed_at] * len(response_onsets))
        conditions.append('buttonpress')
    # get confound regressors and reshape them
    confounds_df = pd.read_csv(confounds_tsv, sep="\t")
    confounds_df = confounds_df[list(confound_names)]
    regressors = confounds_df.values.tolist()
    regressors = np.array(regressors).T.tolist()
    # keep the confound names?
    if not keep_confound_names:
        regressor_names = None
    else:
        regressor_names = list(confound_names)
    # pack in bunch
    design_info = Bunch(conditions=conditions, onsets=onsets, durations=durations,
                        regressor_names=regressor_names, regressors=regressors)
    return design_info


def make_contrasts(design_info):
    """
    Look at all specified conditions (in first run) and set up a one-sample t-test for all of them.
    """
    firstbunch = design_info[0]
    contrasts = []
    for target_cond in firstbunch.conditions:
        contrast = ['{}_contrast'.format(target_cond), 'T']
        cvec_names = [target_cond]
        cvec_vals = [1]
        for nontarget_cond in firstbunch.conditions:
            if target_cond == nontarget_cond:
                continue
            else:
                cvec_names.append(nontarget_cond)
                cvec_vals.append(0)
        contrast.append(cvec_names)
        contrast.append(cvec_vals)
        contrasts.append(contrast)
    # contrasts = [['{}_contrast'.format(condition), 'T', [condition], [1]]
    #              for condition in firstbunch.conditions]
    return contrasts


def sort_copes(copes,
               varcopes,
               contrasts):
    """
    Take output from L1 model fit and reshape / sort it for L2 FFX model.
    """
    import numpy as np
    if not isinstance(copes, list):
        copes = [copes]
        varcopes = [varcopes]
    num_copes = len(contrasts)
    n_runs = len(copes)
    all_copes = np.array(copes).flatten()
    all_varcopes = np.array(varcopes).flatten()
    outcopes = all_copes.reshape(len(all_copes) / num_copes, num_copes).T.tolist()
    outvarcopes = all_varcopes.reshape(len(all_varcopes) / num_copes, num_copes).T.tolist()
    num_copes = len(outcopes)
    return outcopes, outvarcopes, n_runs, num_copes


def create_main_wf(sub='01',
                   ses='1',
                   bids_dir='/home/data/exppsy/oliver/RIRSA/scratch/BIDS',
                   fmriprep_out_dir='/home/data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out',
                   fmrilogs_dir='/home/data/exppsy/oliver/RIRSA/raw/data/fmri_logs',
                   work_base_dir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/pre_glm/workdir',
                   results_basedir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/pre_glm/results',
                   hpcutoff=120.,
                   tr=2,
                   n_runs=4,
                   tr_units='secs',
                   confnames=('a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
                              'a_comp_cor_04', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'),
                   sortconds=True,
                   use_derivatives=True,
                   model_serial_corrs=True,
                   l1_f_contrasts=False,
                   film_thresh=0.,
                   l2_flameo_runmode='fe'):  # ('fe' or 'ols' or 'flame1' or 'flame12')
    """
    Generate a nipype workflow that does the GLM to model stimulus responses
    and aggregate them across runs in a session for a given subject.
    """
    import os
    from os.path import join as pjoin

    # set up workflow with working and results directory
    workflow = pe.Workflow(name='pre_rsa_glm')
    results_subdir = pjoin(results_basedir, sub, ses)
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)
    work_sub_dir = pjoin(work_base_dir, sub, ses)
    if not os.path.exists(work_sub_dir):
        os.makedirs(work_sub_dir)
    workflow.base_dir = work_sub_dir

    # Grab file paths
    datagrabber = pe.Node(name='data_grabber', interface=Function(
        input_names=['sub', 'ses', 'bidsdir', 'fmriprepoutdir', 'fmrilogsdir', 'nruns'],
        output_names=['bold_runs', 'design_tsv_zips', 'bold_masks_per_run', 'anat_mask'], function=grab_data))
    datagrabber.inputs.sub = sub
    datagrabber.inputs.ses = ses
    datagrabber.inputs.bidsdir = bids_dir
    datagrabber.inputs.fmriprepoutdir = fmriprep_out_dir
    datagrabber.inputs.fmrilogsdir = fmrilogs_dir
    datagrabber.inputs.nruns = n_runs

    # apply brain mask to preprocessed bold file (both are already in anatomical space and output of fmriprep)
    applymask = pe.MapNode(name='applymask', interface=ApplyMask(), iterfield=['in_file', 'mask_file'])

    # Create model design for SpecifyModel from tsv files
    make_evs = pe.MapNode(name='make_evs', interface=Function(
        input_names=['design_tsv_zip', 'sort_events', 'confound_names'],
        output_names=['design_info'], function=event_tsvs_to_bunch),
                          iterfield=['design_tsv_zip'])
    make_evs.inputs.sort_events = sortconds
    make_evs.inputs.confound_names = confnames

    # Set up L1 model
    modelspec = pe.Node(interface=SpecifyModel(), name="modelspec")
    modelspec.inputs.input_units = tr_units
    modelspec.inputs.high_pass_filter_cutoff = hpcutoff
    modelspec.inputs.time_repetition = tr

    # make list of contrasts
    contrastgen = pe.Node(name='contrastgen', interface=Function(
        input_names=['design_info'], output_names=['contrasts'], function=make_contrasts))

    # Fit L1 model
    modelfit = create_modelfit_workflow(name='modelfit_wf', f_contrasts=l1_f_contrasts)
    modelfit.inputs.inputspec.interscan_interval = tr
    modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivatives}}
    modelfit.inputs.inputspec.model_serial_correlations = model_serial_corrs
    modelfit.inputs.inputspec.film_threshold = film_thresh

    # Sort COPES output from L1
    cope_sorter = pe.Node(name='cope_sorter', interface=Function(
        input_names=['copes', 'varcopes', 'contrasts'],
        output_names=['copes', 'varcopes', 'n_runs', 'num_copes'], function=sort_copes))

    # Set up L2-model (combine runs in one session)
    fixed_fx = create_fixed_effects_flow()
    fixed_fx.inputs.flameo.run_mode = l2_flameo_runmode

    # dump results
    datasink = pe.Node(interface=DataSink(), name="datasink")
    datasink.inputs.base_directory = results_subdir
    # Connect
    workflow.connect([(datagrabber, applymask, [('bold_runs', 'in_file'),
                                                ('bold_masks_per_run', 'mask_file')])])
    workflow.connect(applymask, 'out_file', modelspec, 'functional_runs')
    workflow.connect(datagrabber, 'design_tsv_zips', make_evs, 'design_tsv_zip')
    workflow.connect(make_evs, 'design_info', modelspec, 'subject_info')
    # to modelfit
    workflow.connect(make_evs, 'design_info', contrastgen, 'design_info')
    workflow.connect(contrastgen, 'contrasts', modelfit, 'inputspec.contrasts')
    workflow.connect(modelspec, 'session_info', modelfit, 'inputspec.session_info')
    workflow.connect(applymask, 'out_file', modelfit, 'inputspec.functional_data')
    # L1 and L2 models
    workflow.connect(contrastgen, 'contrasts', cope_sorter, 'contrasts')
    workflow.connect(datagrabber, 'anat_mask', fixed_fx, 'flameo.mask_file')
    workflow.connect([(modelfit, cope_sorter, [('outputspec.copes', 'copes'),
                                               ('outputspec.varcopes', 'varcopes')]),
                      (cope_sorter, fixed_fx, [('copes', 'inputspec.copes'),
                                               ('varcopes', 'inputspec.varcopes'),
                                               ('n_runs', 'l2model.num_copes')]),
                      (modelfit, fixed_fx, [('outputspec.dof_file',
                                             'inputspec.dof_files')])])
    """
    Sink data
    """
    # modelgen
    workflow.connect([(modelfit.get_node('modelgen'), datasink,
                       [('design_cov', 'qa.model'),
                        ('design_image', 'qa.model.@matrix_image'),
                        ('design_file', 'qa.model.@matrix')])])
    # L2 results
    workflow.connect([(fixed_fx.get_node('outputspec'), datasink,
                       [('res4d', 'res4d'),
                        ('copes', 'copes'),
                        ('varcopes', 'varcopes'),
                        ('zstats', 'zstats'),
                        ('tstats', 'tstats')])])

    return workflow


if __name__ == '__main__':
    # TODO: make run options variable with argparse so runscript.sh can define all parameters

    # from nipype import config, logging
    import sys

    #
    # config.enable_debug_mode()
    # logging.update_logging(config)

    subject = sys.argv[1]
    session = sys.argv[2]

    wf = create_main_wf(sub=subject, ses=session)
    wf.write_graph()
    wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
