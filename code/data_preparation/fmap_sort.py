#!/usr/bin/env python
# coding: utf-8

"""
Execute this script to sort, rescale, and rename the field maps returned from the Philipps Archieva scanner.

Make sure you set up the following variables correctly at the head of the script:
subject_list
session_list
bids_directory
datasink_dir
workdir

The resulting nifti and json files will be stored in the datasink
"""

import argparse
import os
import subprocess
from os.path import join as pjoin

import nipype.algorithms.misc as misc
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from nipype import Workflow
from nipype.interfaces.base import Bunch, BaseInterfaceInputSpec, File, traits, \
    TraitedSpec, BaseInterface
from nipype.interfaces.utility import Function, Rename
from nipype.pipeline import Node, MapNode, Workflow
from nipype.workflows.dmri.fsl.artifacts import all_fmb_pipeline

"""
Set up subject and session lists and directory paths
"""

subject_list = ['%02d' % i for i in xrange(1, 7)]
session_list = ['1', '2']  # TODO: avoid hard coding here
bids_directory = '/home/my_data/exppsy/oliver/RIRSA/scratch/BIDS'
datasink_dir = '/home/my_data/exppsy/oliver/RIRSA/scratch/fmap_sort/datasink'
workdir = "/home/my_data/exppsy/oliver/RIRSA/scratch/fmap_sort/workdir"

"""
Set up FSL
"""
# source /etc/fsl/fsl.sh
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

"""
Fieldmap

Rate den richtigen Speichernamen von Magnitude und Phase.
dcm2nii erzeugt mag/phasen Bilder in unterschiedlicher Reihenfolge.
Gebe Magnitude, Phase in richtiger Reihenfolge zurück.

Wird für Verzerrungskorrektur benötigt.
fieldmap MUSS aus Dicoms erzeugt werden.
"""


class FieldMapInputSpec(BaseInterfaceInputSpec):
    fmap_0 = File(exists=True, desc='first fieldmap file', mandatory=True)
    fmap_0_json = File(exists=True, desc='json to first fieldmap file', mandatory=True)
    fmap_1 = File(exists=True, desc='second fieldmap file', mandatory=True)
    fmap_1_json = File(exists=True, desc='json to secondfieldmap file', mandatory=True)


class FieldMapOutputSpec(TraitedSpec):
    fmap_mag = File(exists=True, desc="magnitude image")
    fmap_mag_json = File(exists=True, desc="magnitude image json")
    fmap_phs = File(exists=True, desc="phasediff image")
    fmap_phs_json = File(exists=True, desc="phasediff image json")


class FieldMap(BaseInterface):
    input_spec = FieldMapInputSpec
    output_spec = FieldMapOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        data = nib.load(self.inputs.fmap_0)
        if int(data.dataobj.inter) == -217:
            self.fmap_mag = self.inputs.fmap_1
            self.fmap_mag_json = self.inputs.fmap_1_json
            self.fmap_phs = self.inputs.fmap_0
            self.fmap_phs_json = self.inputs.fmap_0_json
        else:
            self.fmap_mag = self.inputs.fmap_0
            self.fmap_mag_json = self.inputs.fmap_0_json
            self.fmap_phs = self.inputs.fmap_1
            self.fmap_phs_json = self.inputs.fmap_1_json
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['fmap_mag'] = self.fmap_mag
        outputs['fmap_mag_json'] = self.fmap_mag_json
        outputs['fmap_phs'] = self.fmap_phs
        outputs['fmap_phs_json'] = self.fmap_phs_json
        return outputs


fmap_order = pe.Node(interface=FieldMap(), name='fmap_order')

"""
Skaliere Phasenbild der fieldmap
Wird für Verzerrungskorrektur benötigt

FSL benötigt Phasenbild auf 2 Pi skaliert.
D.h. Radian / delta TE (2.3ms)
Teiler ist 1s / Delta_TE(fieldmap)
Am Philips 3T = 2.3 ms
1/0.0023 = 434.7826
"""

phsscale = pe.Node(interface=fsl.maths.MathsCommand(args='-div 434.7826\
    -mul 6.2831'), name='phs_scale')

# info source
# Map field names to individual subject sessions
info = dict(fmap1=[['subject', 'session']],
            fmap1_json=[['subject', 'session']],
            fmap2=[['subject', 'session']],
            fmap2_json=[['subject', 'session']])
infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id', 'session_id']),
                     name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('session_id', session_list)]

# datasource
datasource = Node(nio.DataGrabber(infields=['subject', 'session'],
                                  outfields=['fmap1', 'fmap2', 'fmap1_json', 'fmap2_json']),
                  name='datagrabber')
datasource.inputs.base_directory = bids_directory
datasource.inputs.template = '*'
datasource.inputs.field_template = {'fmap1': 'sub-%s/ses-%s/fmap/*_fmap1.nii.gz',
                                    'fmap1_json': 'sub-%s/ses-%s/fmap/*_fmap1.json',
                                    'fmap2': 'sub-%s/ses-%s/fmap/*_fmap2.nii.gz',
                                    'fmap2_json': 'sub-%s/ses-%s/fmap/*_fmap2.json'}
datasource.inputs.sort_filelist = True
datasource.inputs.template_args = info

# set up my_data sink
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = datasink_dir

"""
Rename resulting field map images and jsons
"""

rename_mag = pe.Node(Rename(format_string='sub-%(subject)s_ses-%(session)s_magnitude.nii.gz'),
                     name='rename-magnitude')
rename_mag_json = pe.Node(Rename(format_string='sub-%(subject)s_ses-%(session)s_magnitude.json'),
                          name='rename-magnitude-json')
rename_phdiff = pe.Node(Rename(format_string='sub-%(subject)s_ses-%(session)s_fieldmap.nii.gz'),
                        name='rename-rescaled-phasediff')
rename_phdiff_json = pe.Node(Rename(format_string='sub-%(subject)s_ses-%(session)s_fieldmap.json'),
                             name='rename-rescaled-phasediff-json')

"""
Set up workflow
"""

# init workflow
main = Workflow(name="fmap_sort_wf",
                base_dir=workdir)
# info and my_data grabber
main.connect(infosource, 'subject_id', datasource, 'subject')
main.connect(infosource, 'session_id', datasource, 'session')

# Fieldmap sortieren
main.connect(datasource, 'fmap1', fmap_order, 'fmap_0')
main.connect(datasource, 'fmap2', fmap_order, 'fmap_1')
main.connect(datasource, 'fmap1_json', fmap_order, 'fmap_0_json')
main.connect(datasource, 'fmap2_json', fmap_order, 'fmap_1_json')

# Phase skalieren
main.connect(fmap_order, 'fmap_phs', phsscale, 'in_file')

# renaming images
main.connect(infosource, 'subject_id', rename_mag, 'subject')
main.connect(infosource, 'session_id', rename_mag, 'session')
main.connect(fmap_order, 'fmap_mag', rename_mag, 'in_file')
main.connect(infosource, 'subject_id', rename_phdiff, 'subject')
main.connect(infosource, 'session_id', rename_phdiff, 'session')
main.connect(phsscale, 'out_file', rename_phdiff, 'in_file')

# renaming jsons
main.connect(infosource, 'subject_id', rename_mag_json, 'subject')
main.connect(infosource, 'session_id', rename_mag_json, 'session')
main.connect(fmap_order, 'fmap_mag_json', rename_mag_json, 'in_file')
main.connect(infosource, 'subject_id', rename_phdiff_json, 'subject')
main.connect(infosource, 'session_id', rename_phdiff_json, 'session')
main.connect(fmap_order, 'fmap_mag_json', rename_phdiff_json, 'in_file')

# my_data sink
main.connect(fmap_order, 'fmap_mag', datasink, 'fmap_order.fmap_mag')
main.connect(fmap_order, 'fmap_phs', datasink, 'fmap_order.fmap_phs')
main.connect(rename_mag, 'out_file', datasink, 'rename.fmap_mag')
main.connect(rename_mag_json, 'out_file', datasink, 'rename.fmap_mag.json')
main.connect(rename_phdiff, 'out_file', datasink, 'rename.fmap_phs_res')
main.connect(phsscale, 'out_file', datasink, 'phsscale.fmap_phs_res')
main.connect(rename_phdiff_json, 'out_file', datasink, 'rename.fmap_phs_res.json')

if __name__ == '__main__':
    # main.write_graph(graph2use='orig', simple_form=False)
    # main.run(plugin='MultiProc', plugin_args={'n_procs' : 30})
    # main.run(plugin='Condor')
    main.run()
