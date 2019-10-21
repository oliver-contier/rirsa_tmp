#!/usr/bin/env python

"""
Do postprocessing (z-scoring, spatial smoothing, and transform to MNI)
on the similarity maps resulting from rsa_correlation.py
"""

import os

import nipype.pipeline.engine as pe
from nipype.interfaces.fsl import ImageMaths, IsotropicSmooth
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.ants.resampling import ApplyTransforms


def grab_data(sub,
              ses,
              model_names=('cat', 'sem', 'V1', 'V2', 'V4', 'IT'),
              rsa_correlation_workdir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/rsa_correlation/workdir',
              fmriprep_outdir='/home/data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out/fmriprep',
              nruns=4):
    from os.path import join as pjoin
    # grab similarity maps as simple list
    sim_maps = [pjoin(
        rsa_correlation_workdir, sub, ses, str(run), '{}_{}_{}_run-{}.nii.gz'.format(param, training, modelname, run))
        for param in ['p']
        for training in ['trained', 'untrained']
        for modelname in model_names
        for run in range(1, nruns + 1)]

    # grab transformation matrices
    anat2mni_mat = [pjoin(fmriprep_outdir,
                          'sub-{}/anat/sub-{}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'.format(sub, sub))]
    return sim_maps, anat2mni_mat


def create_postrsa_wf(sub='01',
                      ses='1',
                      results_basedir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/zscore_smoothe/results',
                      work_base_dir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/zscore_smoothe/workdir',
                      rsacorrelationworkdir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/rsa_correlation/workdir',
                      fmriprep_outdir='/home/data/exppsy/oliver/RIRSA/scratch/fmriprep/fmriprep_out/fmriprep',
                      smoothe_fwhm=6,
                      mni_template_path='/home/data/exppsy/oliver/RIRSA/raw/mni_template/mni_icbm152_nlin_asym_09c/'
                                         'mni_icbm152_t1_brain.nii.gz',
                      mni_brain_mask='/home/data/exppsy/oliver/RIRSA/raw/mni_template/mni_icbm152_nlin_asym_09c/'
                                      'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'):
    from os.path import join as pjoin

    # set up workflow and directories
    workflow = pe.Workflow(name='post_rsa')
    results_subdir = pjoin(results_basedir, sub, ses)
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)
    work_sub_dir = pjoin(work_base_dir, sub, ses)
    if not os.path.exists(work_sub_dir):
        os.makedirs(work_sub_dir)
    workflow.base_dir = work_sub_dir

    # grab paths to similarity maps from rsa_correlation
    datagrabber = pe.Node(name='data_grabber', interface=Function(
        input_names=['sub', 'ses', 'model_names', 'rsa_correlation_workdir', 'fmriprep_outdir', 'nruns'],
        output_names=['sim_maps', 'anat2mni_mat'], function=grab_data))
    datagrabber.inputs.rsa_correlation_workdir = rsacorrelationworkdir
    datagrabber.inputs.fmriprep_outdir = fmriprep_outdir
    datagrabber.inputs.sub = sub
    datagrabber.inputs.ses = ses
    datagrabber.inputs.nruns = 4
    datagrabber.inputs.model_names = ('cat', 'sem', 'V1', 'V2', 'V4', 'IT')

    # transform p to z maps
    ptoz = pe.MapNode(name='ptoz',
                      interface=ImageMaths(op_string='-ptoz', suffix='_zval'),
                      iterfield=['in_file'])

    # smooth z maps
    smoother = pe.MapNode(name='smoothing',
                          interface=IsotropicSmooth(fwhm=smoothe_fwhm),
                          iterfield=['in_file'])

    # apply transformation to MNI
    applytrans = pe.MapNode(name='applytrans',
                            interface=ApplyTransforms(reference_image=mni_template_path),
                            iterfield=['input_image'])

    # mask transformed file again with mni mask to get rid of smoothed out edges
    applymask = pe.MapNode(name='applymask',
                           interface=ApplyMask(mask_file=mni_brain_mask),
                           iterfield=['in_file'])

    # datasink
    datasink = pe.Node(interface=DataSink(base_directory=results_subdir), name="datasink")

    # connect nodes
    workflow.connect(datagrabber, 'sim_maps', ptoz, 'in_file')
    workflow.connect(ptoz, 'out_file', smoother, 'in_file')
    workflow.connect(ptoz, 'out_file', datasink, 'zscored')
    workflow.connect(smoother, 'out_file', datasink, 'smoothe')
    workflow.connect(smoother, 'out_file', applytrans, 'input_image')
    workflow.connect(datagrabber, 'anat2mni_mat', applytrans, 'transforms')
    workflow.connect(applytrans, 'output_image', applymask, 'in_file')
    workflow.connect(applymask, 'out_file', datasink, 'transformed')

    return workflow


if __name__ == '__main__':
    import sys

    subj = sys.argv[1]
    sess = sys.argv[2]

    wf = create_postrsa_wf(sub=subj, ses=sess)
    wf.run()
