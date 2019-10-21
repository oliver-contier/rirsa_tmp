#!/usr/bin/env python

"""
TODO: description

# TODO: this is deprecated. Look at my newer script with contrast coded regressors in design.
"""

# config.enable_debug_mode()

import os
from os.path import join as pjoin

import nipype.pipeline.engine as pe
from nipype.interfaces.fsl import Merge
from nipype.interfaces.fsl import MultipleRegressDesign
from nipype.interfaces.fsl.model import Randomise
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function


def grab_postrsa_output(zscoresmooth_results='/home/data/exppsy/oliver/RIRSA/scratch/rsa/zscore_smoothe/results',
                        model_name='cat',
                        nsubs=6):
    """
    return list of z maps for a given model, but all subjects, in the order that is required by my GLM
    """
    from os.path import join as pjoin
    import glob
    images = [
        glob.glob(pjoin(
            zscoresmooth_results, '{:02d}'.format(sub), '{}'.format(ses), 'transformed', '_applymask*',
            'p_{}_{}_run-{}_zval_smooth_trans_masked.nii.gz'.format(training, model_name, run)
        ))
        for sub in range(1, nsubs + 1)
        for ses in [1, 2]
        for training in ['trained', 'untrained']
        for run in range(1, 5)
    ]
    # flatten list because glob causes nesting
    images = [item for sublist in images for item in sublist]

    return images


def rirsa_contrasts(nsubs=6):
    """
    Return a list of contrasts for use in FSLs randomise.
    """
    # regressor names must include individual subject regressors
    regressor_names = ['train1', 'train2', 'untrain1', 'untrain2'] + \
                      ['sub{:02d}'.format(sub) for sub in range(1, nsubs + 1)]

    # list of contrasts
    contrast_list = [
        ('intercept', 'T', regressor_names, [1, 1, 1, 1] + [0] * nsubs),
        # main effects (not so interesting for our hypotheses)
        ('main_train_untrain', 'T', regressor_names, [1, 1, -1, -1] + [0] * nsubs),
        ('main_untrain_train', 'T', regressor_names, [-1, -1, 1, 1] + [0] * nsubs),
        ('main_post_pre', 'T', regressor_names, [-1, 1, -1, 1] + [0] * nsubs),
        ('main_pre_post', 'T', regressor_names, [1, -1, 1, -1] + [0] * nsubs),
        # various kinds of interactions
        # reversal effects assume that the direction of training effect flips from one session to another
        ('pos_reversal', 'T', regressor_names, [-1, 1, 1, -1] + [0] * nsubs),
        ('neg_reversal', 'T', regressor_names, [1, -1, -1, 1] + [0] * nsubs),
        # postonly assumes that trainied stimuli in session 2 show greater effect than all other conditions
        ('pos_postonly', 'T', regressor_names, [-1, 3, -1, -1] + [0] * nsubs),
        ('neg_postonly', 'T', regressor_names, [1, -3, 1, 1] + [0] * nsubs),
        # increase assumes that both sessions show training effect, but one is larger than the other
        ('increase_training', 'T', regressor_names, [1, 3, -2, -2] + [0] * nsubs),
        ('decrease_training', 'T', regressor_names, [3, 1, -2, -2] + [0] * nsubs),
        # use these for manual conjunction / difference analysis
        ('seperate_pre', 'T', regressor_names, [1, 0, -1, 0] + [0] * nsubs),
        ('seperate_post', 'T', regressor_names, [0, 1, 0, -1] + [0] * nsubs)
    ]
    return contrast_list


def rirsa_make_design(nsubs=6):
    """
    create a dict that contains one key-value pair for each regressor,
    and a list of tuples that represent our wanted contrasts
    the result can be used in FSL's MultipleRegressionDesign
    """
    # regressors for one subject
    train1 = [1, 1, 1, 1] + [0] * 12
    train2 = [0] * 8 + [1, 1, 1, 1, 0, 0, 0, 0]
    untrain1 = [0, 0, 0, 0, 1, 1, 1, 1] + [0] * 8
    untrain2 = [0] * 12 + [1, 1, 1, 1]

    # create design dict which contains each regressor repeated for each subject
    design_dict = {'train1': train1 * nsubs,
                   'train2': train2 * nsubs,
                   'untrain1': untrain1 * nsubs,
                   'untrain2': untrain2 * nsubs}

    # create one regressor for each subject, add to design_dict
    for subint in range(1, nsubs + 1):
        substring = 'sub{:02d}'.format(subint)
        # list of zeros before subject occurs, ones whenever subject occurs
        design_dict[substring] = [0] * ((subint - 1) * 16) + [1] * 16
        # padding zeros until required length
        design_dict[substring] += [0] * ((nsubs * 16) - len(design_dict[substring]))

    # list of subject identifying integers [1,1,1,1,...,2,2,2,2,...]
    subject_groups = [subint for subint in range(1, nsubs + 1) for _ in range(16)]
    assert len(subject_groups) == nsubs * 16

    return design_dict, subject_groups


def create_group_glm_wf(n_subs=6,
                        modelname='cat',
                        contrast_number=0,
                        work_basedir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/rsacorr_groupstats/work_basedir',
                        results_basedir='/home/data/exppsy/oliver/RIRSA/scratch/rsa/rsacorr_groupstats/results_basedir',
                        mnimask='/home/data/exppsy/oliver/RIRSA/raw/mni_template/mni_icbm152_nlin_asym_09c/'
                                'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii',
                        nperm=1000,
                        randseed=42,
                        voxpvalues=False,
                        demean_design=False,
                        do_tfce=True,
                        tfceh=2,
                        tfcee=.5,
                        variance_smoothing_mm=6,
                        only_info_parallel=False,
                        output_raw_stats=True,
                        output_nulldisttxt=True):
    """
    # TODO: docstring
    FSL documentation strongly recommends leaving the tfce height and extent parameters at their default values.
    """
    assert modelname in ['cat', 'sem', 'V1', 'V2', 'V4', 'IT']

    # pick contrast for this parallellization
    all_contrasts = rirsa_contrasts(n_subs)
    chosen_contrast = all_contrasts[contrast_number]

    # create directories
    workdir = pjoin(work_basedir, modelname, 'nperm-{}'.format(nperm), 'contrast-{}'.format(chosen_contrast[0]))
    resultsdir = pjoin(results_basedir, modelname, 'nperm-{}'.format(nperm), 'contrast-{}'.format(chosen_contrast[0]))
    for dir_ in [workdir, resultsdir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # initiate workflow
    workflow = pe.Workflow(name='group_permutation_glm')
    workflow.base_dir = workdir

    # get postprocessed RSA output images for a given model
    grabimages = pe.Node(name='grabimages', interface=Function(
        function=grab_postrsa_output,
        input_names=['zscoresmooth_results', 'model_name', 'nsubs'], output_names=['images']))
    grabimages.inputs.nsubs = n_subs
    grabimages.inputs.model_name = modelname

    makedesign = pe.Node(name='makedesign', interface=Function(
        function=rirsa_make_design,
        input_names=['nsubs'], output_names=['design_dict', 'subject_groups']))
    makedesign.inputs.nsubs = n_subs

    # Generate design for GLM/ randomise
    model = pe.Node(name='model', interface=MultipleRegressDesign(contrasts=[chosen_contrast]))

    # merge z-maps into a 4D image which serves as data for GLM / randomise.
    copemerge = pe.Node(name='copemerge', interface=Merge(dimension='t'))

    # set up permutation GLM
    randomise = pe.Node(name='randomise', interface=Randomise(
        num_perm=nperm, seed=randseed, vox_p_values=voxpvalues, mask=mnimask, tfce=do_tfce,
        show_info_parallel_mode=only_info_parallel, raw_stats_imgs=output_raw_stats,
        p_vec_n_dist_files=output_nulldisttxt, demean=demean_design, var_smooth=variance_smoothing_mm))
    if do_tfce:
        randomise.inputs.tfce_H = tfceh
        randomise.inputs.tfce_E = tfcee

    # datasink
    datasink = pe.Node(interface=DataSink(base_directory=resultsdir), name="datasink")

    # connect
    workflow.connect(grabimages, 'images', copemerge, 'in_files')
    workflow.connect(makedesign, 'design_dict', model, 'regressors')
    workflow.connect(makedesign, 'subject_groups', model, 'groups')
    workflow.connect(model, 'design_grp', randomise, 'x_block_labels')
    workflow.connect(copemerge, 'merged_file', randomise, 'in_file')
    workflow.connect(model, 'design_mat', randomise, 'design_mat')
    workflow.connect(model, 'design_con', randomise, 'tcon')
    workflow.connect(randomise, 'tstat_files', datasink, 'tstat')
    workflow.connect(randomise, 't_corrected_p_files', datasink, 'tfce')

    return workflow


if __name__ == '__main__':
    import sys

    model = sys.argv[1]
    contrast = int(sys.argv[2])-1
    wf = create_group_glm_wf(modelname=model,
                             contrast_number=contrast,
                             nperm=1000,
                             output_raw_stats=True,
                             voxpvalues=True)
    wf.run()
