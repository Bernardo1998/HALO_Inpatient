# HALO
[![DOI](https://zenodo.org/badge/627704812.svg)](https://zenodo.org/badge/latestdoi/627704812)

This is the source code for reproducing the inpatient dataset experiments found in the paper "Synthesizing Extremely High Dimensional Electronic Health Records."

## Setup the environment:
```
pip install -r requirements.txt
```

## Converting the dvlog to suitable format

First please create a "dvlog" folder and place "visual.csv" and acoustic.csv under it.

```
python build_acoustic_dataset.py
python build_visual_dataset.py
```

Once completed, there should be two folders "acoustic" and "visual" under the "dvlog" folder.

## Training a model

First create a "save" folder, then run:

```
python train_model_acoustic.py
python train_model_visual.py
```

Once completed, the trained models should be saved under /save. 

## Generating the Dataset

Generate synthetic dataset by:

```

python test_model_acoustic.py
python test_model_visual.py
```

Once generation is completed, the synthetic dataset should be under results/acousticsets and results/visualsets in .pkl format. 
