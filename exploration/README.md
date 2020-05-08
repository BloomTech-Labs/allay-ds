# Overview

This folder contains notebooks and files used to explore and load the data
and to train and evaluate models.

# Instructions

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

Availalbe sweeps:
- sweep.yaml : sweep_train.py
- rnn_sweep.yaml : rnn_sweep_train.py
- sweep_cnn.yaml : sweep_cnn_train.py
