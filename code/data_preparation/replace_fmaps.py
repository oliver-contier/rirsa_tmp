#! /usr/env/bin/python

import os
import glob
import shutil
import json
from os.path import join as pjoin
from shutil import copyfile

sorting_datasink = '/home/my_data/exppsy/oliver/RIRSA/scratch/fmap_sort/datasink'
bids_basedir = '/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS'

"""
# TODO: Bids Validator wants EchoTime1 and EchoTime2 ...
# according to Philipps hand book, TE = Delta-TE = 2.29999995
# Act. TR/TE (ms) =		"848 / 2.3"
# Min. TR/TE (ms) =		"848 / 1.57"

http://macduff.usc.edu/ee591/library/Pauly-FieldMaps.pdf

There are many different ways that field maps can
be acquired. A simple method is to collect images
at two different echo times.

... the change in echo times is delta TE = TE2-TE1
For now, I'll just add EchoTime1 = 2.29999995 and see what bids validator sais
"""

subjects = ['%02d' % i for i in xrange(1, 6 + 1)]
sessions = ['1', '2']
bids_basedir = '/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS'
scratchdir = '/home/my_data/exppsy/oliver/RIRSA/scratch'

for subject in subjects:
    for session in sessions:
        # source image paths
        mag_img = pjoin(sorting_datasink, 'rename', 'fmap_mag',
                        '_session_id_%s_subject_id_%s' % (session, subject),
                        'sub-%s_ses-%s_magnitude.nii.gz' % (subject, session))
        mag_json = pjoin(sorting_datasink, 'rename', 'fmap_mag', 'json',
                         '_session_id_%s_subject_id_%s' % (session, subject),
                         'sub-%s_ses-%s_magnitude.json' % (subject, session))
        ph_img = pjoin(sorting_datasink, 'rename', 'fmap_phs_res',
                       '_session_id_%s_subject_id_%s' % (session, subject),
                       'sub-%s_ses-%s_fieldmap.nii.gz' % (subject, session))
        ph_json = pjoin(sorting_datasink, 'rename', 'fmap_phs_res', 'json',
                        '_session_id_%s_subject_id_%s' % (session, subject),
                        'sub-%s_ses-%s_fieldmap.json' % (subject, session))

        # copy to destination
        for sourcefile in [mag_img, mag_json, ph_img, ph_json]:
            # destination file name
            destinationfile = pjoin(bids_basedir, 'sub-%s' % subject, 'ses-%s' % session, 'fmap',
                                    sourcefile.split('/')[-1])
            copyfile(sourcefile, destinationfile)

        """
        move old field_maps (which came unsorted from scanner) to a scratch directory
        """
        old_fmaps = glob.glob(pjoin(bids_basedir, 'sub-%s' % subject, 'ses-%s' % session,
                                    'fmap', 'sub-%s_ses-%s_fmap*' % (subject, session)))
        scratch_subdir = pjoin(scratchdir, 'fmap_sort', 'old_fmaps', 'sub-%s' % subject, 'ses-%s' % session, 'fmap')
        if not os.path.exists(scratch_subdir):
            os.makedirs(scratch_subdir)
        for old_fmap in old_fmaps:
            if os.path.exists(old_fmap):
                old_fmap_backup = pjoin(scratch_subdir, old_fmap.split('/')[-1])
                shutil.move(old_fmap, old_fmap_backup)
