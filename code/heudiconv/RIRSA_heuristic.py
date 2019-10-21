import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    # anatomical
    t1w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')

    # field map
    # fmap_ph = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_phasediff')
    # fmap_mag = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_magnitude')
    fmap = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_fmap')

    # RI only runs
    bold_ri_run1 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-ri_run-1_bold')
    bold_ri_run2 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-ri_run-2_bold')
    bold_ri_run3 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-ri_run-3_bold')
    bold_ri_run4 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-ri_run-4_bold')

    # all objects runs
    bold_all_run1 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-all_run-1_bold')
    bold_all_run2 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-all_run-2_bold')
    bold_all_run3 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-all_run-3_bold')

    info = {
        t1w: [],
        fmap: [],
        bold_ri_run1: [],
        bold_ri_run2: [],
        bold_ri_run3: [],
        bold_ri_run4: [],
        bold_all_run1: [],
        bold_all_run2: [],
        bold_all_run3: []
    }

    for idx, s in enumerate(seqinfo):
        # anatomical image
        if 'T1W' in s.protocol_name:
            info[t1w].append(s.series_id)
        # field map phase diff image
        if 'field map_check' in s.protocol_name:
            info[fmap].append(s.series_id)
        # bold images
        if 'EPI_3mm' in s.protocol_name:
            run_id = s.protocol_name.split('_')[-1]
            # run 1 to 4 belongs to task ri
            if run_id == 'run1':
                info[bold_ri_run1].append(s.series_id)
            if run_id == 'run2':
                info[bold_ri_run2].append(s.series_id)
            if run_id == 'run3':
                info[bold_ri_run3].append(s.series_id)
            if run_id == 'run4':
                info[bold_ri_run4].append(s.series_id)
            # run 3 to 7 belong to task all
            if run_id == 'run5':
                info[bold_all_run1].append(s.series_id)
            if run_id == 'run6':
                info[bold_all_run2].append(s.series_id)
            if run_id == 'run7':
                info[bold_all_run3].append(s.series_id)

    return info
