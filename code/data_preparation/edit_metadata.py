#!/usr/bin/env python

import glob
import json
from os.path import join as pjoin

import numpy as np


def edit_json_field(jsonfile, fieldname, value):
    """
    Load json file, write vaue to fieldname, and close again.
    """
    with open(jsonfile) as infile:
        content = json.load(infile)
    content[fieldname] = value
    with open(jsonfile, 'w') as outfile:
        json.dump(content, outfile)


def make_slices(tr=2., n_slices=37):
    """
    Create array of equidistant slice times, going through odd slices first,
    and then the even slices.
    """
    slice_indices = np.arange(n_slices)
    slice_times = np.empty(n_slices, dtype=float)
    val = 0.
    for indices_subset in [slice_indices[0::2], slice_indices[1::2]]:
        for idx in indices_subset:
            slice_times[idx] = val
            val += float(tr) / float(n_slices)
    return slice_times


def calc_effective_echo_spacing(bandwidthperpixelphaseencoding,
                                matrixsizephase):
    # TODO: Formula seems correct, but I got the order of magnitude wrong ...
    # result in seconds
    return 1 / (bandwidthperpixelphaseencoding * matrixsizephase)


def edit_metadata(bids_basedir='/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS',
                  subjects=('01', '02'),
                  sessions=('1', '2'),
                  add_rads_unit=True,
                  phase_encoding_dirfromax=True,
                  invert_phase_encoding=False,
                  add_effective_echo_spacing=True,
                  add_slice_times=True,
                  slice_times=make_slices(tr=2., n_slices=37),
                  cogatlasid='http://www.cognitiveatlas.org/task/id/trm_4ebd47b8bab6b/',
                  rename_fmaps_in_scanstsv=True,
                  add_fmap_intended_for=True):
    # TODO: docstring

    """
    start with general metadata
    """

    task_bold_jsons = glob.glob(pjoin(bids_basedir, 'task-*_bold.json'))
    for task_bold_json in task_bold_jsons:
        edit_json_field(task_bold_json, 'MatrixCoilMode', 'SENSE')
        # SENSE accelleration factor 2
        edit_json_field(task_bold_json, 'MultibandAccellerationFactor', 2)
        # CogAtlasID: object one-back task
        edit_json_field(task_bold_json, 'CogAtlasID', cogatlasid)

    """
    Iterate over subjects, sessions and runs
    """
    for subject in subjects:
        for session in sessions:

            """
            Edit json images belonging to fieldmaps
            """

            fmap_json = pjoin(bids_basedir, 'sub-%s' % subject, 'ses-%s' % session, 'fmap',
                              'sub-%s_ses-%s_fieldmap.json' % (subject, session))
            mag_json = pjoin(bids_basedir, 'sub-%s' % subject, 'ses-%s' % session, 'fmap',
                             'sub-%s_ses-%s_magnitude.json' % (subject, session))
            with open(fmap_json) as infile:
                fmap_content = json.load(infile)

            # rad/s as unit in _fieldmap.json

            if add_rads_unit:
                fmap_content['Units'] = 'rad/s'

            # add "IntendedFor" to _fieldmap.json and _magnitude.json
            if add_fmap_intended_for:
                # paths for bold images in first session
                intended_bold_images = ['ses-%s/func/sub-%s_ses-%s_task-ri_run-%i_bold.nii.gz'
                                        % (session, subject, session, run)
                                        for run in range(1, 5)]  # 4 runs in ri-only task
                if session == '2':
                    intended_bold_images += ['ses-%s/func/sub-%s_ses-%s_task-all_run-%i_bold.nii.gz'
                                             % (session, subject, session, run)
                                             for run in range(1, 4)]  # 3 runs in task with all stimuli
                # add to _fieldmap.json and _magnitude.json
                edit_json_field(fmap_json, 'IntendedFor', intended_bold_images)
                edit_json_field(mag_json, 'IntendedFor', intended_bold_images)

            if rename_fmaps_in_scanstsv:
                # edit _scans.tsv to rename fieldmap1 / fieldmap2 into fieldmap / magnitude
                scanstsv = pjoin(bids_basedir, 'sub-%s' % subject, 'ses-%s' % session,
                                 'sub-%s_ses-%s_scans.tsv' % (subject, session))
                with open(scanstsv, 'r') as file:
                    filedata = file.read()
                for old, new in zip(['fmap1', 'fmap2'], ['fieldmap', 'magnitude']):
                    filedata = filedata.replace(old, new)
                with open(scanstsv, 'w') as file:
                    file.write(filedata)

            # edit _bold.json files
            bold_jsons = glob.glob(pjoin(bids_basedir, 'sub-%s' % subject, 'ses-%s' % session,
                                         'func', '*_bold.json'))
            for bold_json in bold_jsons:
                with open(bold_json) as infile:
                    content = json.load(infile)

                # set bold my_data phase encoding direction
                if phase_encoding_dirfromax:
                    if invert_phase_encoding:
                        content['PhaseEncodingDirection'] = '-' + content['PhaseEncodingAxis']
                    else:
                        content['PhaseEncodingDirection'] = content['PhaseEncodingAxis']

                # slice times
                if add_slice_times:
                    content['SliceTiming'] = list(slice_times)

                # set (effective) echo spacing
                if add_effective_echo_spacing:
                    content['EffectiveEchoSpacing'] = .00061
                    # TODO: don't hard code! Read e-mails with joerg again,
                    #  somehow I was wrong with the order of magnitude

                with open(bold_json, 'w') as outfile:
                    json.dump(content, outfile)

    return None


if __name__ == '__main__':
    edit_metadata(subjects=['%02d' % i for i in xrange(1,7)])
