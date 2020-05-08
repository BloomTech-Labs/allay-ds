# Overview

This folder contains notebooks and files used to explore and load the data
and to train and evaluate models.

# Notebooks
- [explore_data.ipynb](explore_data.ipynb)
  - Combine and deduplicte datasets
- [data2lemma2vec.ipynb](data2lemma2vec.ipynb)
  - Convert training dataset to lemmas and vectors
  - Export results as pickled dataframes
- [train_ml_models.ipynb](train_ml_models.ipynb)
  - Baseline traditional machine learning models
- [train_nn_models.ipynb](train_nn_models.ipynb)
  - Baseline neural network models

# WandB Hyperparameter Sweep Instructions

Run the following two lines of code to initialize and run a sweep. The first
line of code will give you a sweep id which you will need to insert in the
second line.
```python
wandb sweep -p allay-ds-23 exploration/<configuration yaml file>
wandb agent USER/PROJECT/SWEEP_ID
```

To tune what hyperparameters are being searched across and the search method,
edit the `sweep.yaml` file. This file currently sweeps across the training
process coded within `sweep_train.py`, but this is configurable as well. If not
logged into weights and biases through the command line, WANDB_API_KEY
must be set as an environment variable within the `sweep_train.py` file
or its equivalent.

Sweeps will be run and logged until the process
is terminated with control-C (or your OS's equivalent).

[Weights and Biases Sweeps](https://docs.wandb.com/sweeps/) contains further
documentation.

### Availalbe sweeps:
- [sweep.yaml](sweep.yaml) : [sweep_train.py](sweep_train.py)
- [rnn_sweep.yaml](rnn_sweep.yaml) : [rnn_sweep_train.py](rnn_sweep_train.py)
- [sweep_cnn.yaml](sweep_cnn.yaml) : [sweep_cnn_train.py](sweep_cnn_train.py)

### Additional files:
- [process_data.py](process_data.py)
  - helper functions to minimize code reuse
