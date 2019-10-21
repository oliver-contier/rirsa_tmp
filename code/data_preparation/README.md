# Preparation of the BIDS dataset

Necessary environment paths:

```
source /etc/fsl/fsl.sh
source /path/to/dataset/raw/venvs/fmap_sort_env/bin/activate
```

Also, you might need to add permissions to write to the BIDS dataset directory

```
chmod -R /path/to/dataset/scratch/BIDS
```

## Add stimulus files

Manually copy both stimulus folders (percepts and preprocessed) into the BIDS/stimuli directory.

## Transform Field Maps

- [fmap_sort.py](fmap_sort.py): The Philipps scanner returns fieldmap and magnitude 
images in ambiguous order. This script reorders them and saves them with correct naming.
- [replace_fmaps.py](replace_fmaps.py): Copy the newly created fieldmap and magnitude files into the BIDS data set.

## Edit and add  metadata

- [edit_metadata.py](edit_metadata.py): Iterate over a selection of json files in the BIDS data set and add information to them:
    - Add the "Unit" (rd/s) key to the _fieldmap.json descriptors.
    - 'MatrixCoilMode': 'SENSE' and 'MultibandAccellerationFactor': 2 in task-*_bold.json
    - 'CogAtlasID' for object one-back task
    - set PhaseEncodingDirection, EffectiveEchoSpacing, and SliceTiming in *_bold.json for each subject, session, and run.
- [log2events.py](log2events.py): Fill in stimulus events into  _event.tsv files.

## TODO: 
- Describe participants.tsv columns in participants.json