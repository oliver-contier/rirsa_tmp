"""
Simple script that produces surface plots for the interaction contrasts from our rsacorr_groupstats.
For each model, hemisphere, and view, a new png file is created.
"""

import os
from os.path import join as pjoin
import matplotlib as mpl; mpl.use('Agg')  # don't use display (bc running on medusa)

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting, surface
from nilearn.image import load_img

# results base directory
results_basedir = '/home/data/exppsy/oliver/RIRSA/scratch/rsa/rsacorr_groupstats/results_basedir'
nperms = 500

# iterate over contrasts
for contrast_subdir, contrast_name, colorscheme in zip(
        ['posttrain_minus_pretrain', 'pretrain_minus_posttrain'],
        ['post-pre', 'pre-post'],
        ['OrRd', 'GnBu']):

    # iterate over RSA model results
    for model_subdir, model_name in zip(['cat', 'sem', 'V1', 'V2', 'V4', 'IT'],
                                        ['categorical', 'semantic', 'V1', 'V2', 'V4', 'IT']):

        # define output directory for png files
        output_dir = pjoin('/home/data/exppsy/oliver/RIRSA/raw/code/RSA/glm_group_perm/plot_results/output',
                           contrast_subdir,
                           model_subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get statistical map
        statsmap_fname = pjoin(results_basedir, model_subdir, 'nperm-{}'.format(nperms),
                               'contrast-{}'.format(contrast_subdir), 'tfce',
                               'randomise_tfce_corrp_tstat1.nii.gz')

        # find out vmax
        maxval = np.max(load_img(statsmap_fname).get_data())
        halfmax = float(maxval) / 2

        # get a cortical mesh
        fsaverage = datasets.fetch_surf_fsaverage()

        # loop over views
        for name_view in ['lateral', 'medial', 'ventral']:
            # loop over hemispheres
            for name_hemi, fs_pial_hemi, fs_sulc_hemi, fs_infl_hemi in zip(['left', 'right'],
                                                                           [fsaverage.pial_left, fsaverage.pial_right],
                                                                           [fsaverage.sulc_left, fsaverage.sulc_right],
                                                                           [fsaverage.infl_left, fsaverage.infl_right]):
                # Sample the statistical map around each node of the mesh
                results_texture = surface.vol_to_surf(statsmap_fname, fs_pial_hemi)

                title_string = '{} model\n{}\n(vmax = {:.3f})'.format(model_name, contrast_name, maxval)

                # plot
                g = plotting.plot_surf_stat_map(fs_infl_hemi, results_texture,
                                                hemi=name_hemi,
                                                title=title_string,
                                                colorbar=True,
                                                cmap=colorscheme,
                                                view=name_view,
                                                threshold=.5,
                                                bg_map=fs_sulc_hemi,
                                                # darkness=.1,
                                                symmetric_cbar=False,
                                                vmax=maxval)

                # save, close, tell me
                g.savefig(pjoin(output_dir, '{}_{}_{}.png'.format(model_subdir, name_view, name_hemi)), dpi=300)
                plt.close()  # clear figure as to not run out of memory
                print('finished {} {} {} {}'.format(contrast_name, model_name, name_view, name_hemi))
