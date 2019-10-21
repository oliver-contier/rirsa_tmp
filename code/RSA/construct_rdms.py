#!/usr/bin/env python

from os.path import join as pjoin

import numpy as np
from nltk.corpus import wordnet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore


def id2english():
    mapper = {'9': 'shoe',
              '40': 'light_bulb',
              '89': 'tube',
              '90': 'duck',
              '125': 'mug',
              '161': 'jug',
              '263': 'frame',
              '281': 'garlic',
              '405': 'puncher',
              '408': 'dice',
              '461': 'toilet_paper',
              '466': 'pen',
              '615': 'can',
              '642': 'stapler',
              '910': 'atomizer',
              '979': 'hat'}
    return mapper


def construct_cornet_rdms(stimlist,
                          cornet_out_dir="/home/my_data/exppsy/oliver/RIRSA/raw/my_data/cornet_output",
                          layers=('V1', 'V2', 'V4', 'IT'),
                          model_type='S',
                          pdist_metric='correlation',
                          zscore_rdms=True,
                          return_triu=True,
                          triu_k=1):
    """
    Given a list of stimuli, construct one model RDM for each CORnet layer used for feature extraction.
    Returns an array of size (n_upper_triu_dissim, n_layers).
    Result should be directly usable for RSA Regression in pymvpa.
    """
    rdms = []
    for layer in layers:
        stim_features = []
        for stim in stimlist:
            # get features
            features_arr = np.load(
                pjoin(cornet_out_dir, stim + '_ri_percept', 'CORnet-{}_{}_output_feats.npy'.format(model_type, layer)))
            stim_features.append(features_arr[0])  # removed empty dimension
        rdm = squareform(pdist(np.array(stim_features), pdist_metric))
        assert np.shape(rdm) == (len(stimlist), len(stimlist))
        if return_triu:
            rdm = rdm[np.triu_indices_from(rdm, k=triu_k)]
        if zscore_rdms:
            rdm = zscore(rdm)
        rdms.append(rdm)
    rdms = np.array(rdms)
    rdms = np.swapaxes(rdms, 0, 1)
    return rdms


def construct_categorical_rdm(obj_list,
                              return_triu=True,
                              z_score=False):
    """
    Given a list of stimuli, construct a categorical RDM with 1 when object ID is the same and 0 elsewhere.
    """
    cat_rdm = []
    for obj1 in obj_list:
        row = []
        for obj2 in obj_list:
            if obj1.split('_')[0] == obj2.split('_')[0]:
                row.append(1.)
            else:
                row.append(0.)
        cat_rdm.append(row)
    cat_rdm = np.array(cat_rdm)
    assert np.shape(cat_rdm) == (64, 64)
    if return_triu:
        cat_rdm = cat_rdm[np.triu_indices_from(cat_rdm, k=1)]
    if z_score:
        cat_rdm = zscore(cat_rdm)
    return cat_rdm


def construct_semantic_rdm(obj_list,
                           id2english_dict=id2english(),
                           return_triu=True,
                           z_score=True,
                           sim_metric='wup'):
    """
    Construct semantic RDM based on wordnet similarity for a list of object_ids.
    Values always denote 1-similarity, i.e. semantic distance.
    Note that lch similarity has a negative sign already, so we don't subtract it.
    """
    sem_rdm = []
    for obj1 in obj_list:
        row = []
        obj1_id = obj1.split('_')[0]
        obj1_name = id2english_dict[obj1_id]
        obj1_syns = wordnet.synsets(obj1_name)[0]  # grab the first synset
        for obj2 in obj_list:
            obj2_id = obj2.split('_')[0]
            obj2_name = id2english_dict[obj2_id]
            obj2_syns = wordnet.synsets(obj2_name)[0]
            if sim_metric == 'path':
                path_sim = 1 - obj1_syns.path_similarity(obj2_syns)
            elif sim_metric == 'wup':
                path_sim = 1 - obj1_syns.wup_similarity(obj2_syns)
            elif sim_metric == 'lch':
                path_sim = obj1_syns.lch_similarity(obj2_syns)
            else:
                raise IOError('Did not recognize your semantic similarity metric: {}'.format(sim_metric))
            row.append(path_sim)
        sem_rdm.append(row)
    sem_rdm = np.array(sem_rdm)
    assert np.shape(sem_rdm) == (64, 64)
    if return_triu:
        sem_rdm = sem_rdm[np.triu_indices_from(sem_rdm, k=1)]
    if z_score:
        sem_rdm = zscore(sem_rdm)
    return sem_rdm


def makemodel_IT_semantic_categorical(cornet_rdms,
                                      semantic_rdm,
                                      categorical_rdm):
    """
    Based on previuosly produced RDMs, make a pymvpa-RSA compatible model matrix
    with shape (n_features, n_predictors).
    Remember that in our actual analysis, we have seperate semantic and cornet rdms for trained and untrained stimuli.
    """
    it_rdm = cornet_rdms[:, -1]
    model_array = np.array([it_rdm, semantic_rdm, categorical_rdm])
    model_array = np.swapaxes(model_array, 0, 1)
    assert np.shape(model_array) == (len(semantic_rdm), 3)
    return model_array
