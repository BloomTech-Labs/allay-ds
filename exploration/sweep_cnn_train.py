"""This module runs the training function that weights and biases fills in
with varying hyperparameters in a hyperparameter sweep.
"""

import os
import pickle

from dotenv import load_dotenv
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from tensorflow.keras.layers import (Concatenate, Conv1D, Dense, Dropout,
                                     Embedding, GlobalMaxPooling1D, Input)
from tensorflow.keras.models import Model

import wandb
from wandb.keras import WandbCallback

# Set defaults for each parameter you are sweeping through.
hyperparameter_defaults = dict(
    learning_rate=0.001,
    epochs=5,
    optimizer="adam",
    num_filters=2,
    base_kernel_size=2,
    dropout_rate=0.3,
    loss_function="binary_crossentropy"
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Import data prepped for training CNN in notebook: train_nn_models.ipynb
with open('exploration/data/cnn_data.pkl', 'rb') as f:
    cnn_data = pickle.load(f)

x_train = cnn_data['x_train']
y_train = cnn_data['y_train']
x_val = cnn_data['x_val']
y_val = cnn_data['y_val']
EMBEDDINGS_LEN = cnn_data['EMBEDDINGS_LEN']
MAX_SEQ_LENGTH = cnn_data['MAX_SEQ_LENGTH']
N_FEATURES = cnn_data['N_FEATURES']
embeddings_index = cnn_data['embeddings_index']


def build_model(num_filters=2, dropout_rate=0.3, base_kernel_size=2):
    """Builds a CNN with hyperparamters.

    :param num_filters: int, default=2. Number of filters in each
    convolutional layer.

    :param droput_rate: float, default=0.3. Dropout rate of each layer.

    :param base_kernel_size: int, default=2. Size of stride of first
    convolutional layer. Each additional layer's stride is incremented from
    this value.

    :returns: Instantiated, uncompiled model.
    """
    # Input Layer
    inputs = Input(shape=(MAX_SEQ_LENGTH,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=N_FEATURES + 1,
                                output_dim=EMBEDDINGS_LEN,
                                # pre-trained embeddings
                                weights=[embeddings_index],
                                input_length=MAX_SEQ_LENGTH,
                                trainable=False,
                                )(inputs)
    embedding_dropped = Dropout(dropout_rate)(embedding_layer)

    # Convolution Layer - 3 Convolutions, each connected to input embeddings
    # Branch a
    conv_a = Conv1D(filters=num_filters,
                    kernel_size=base_kernel_size,
                    activation='relu',
                    )(embedding_dropped)
    pooled_conv_a = GlobalMaxPooling1D()(conv_a)
    pooled_conv_dropped_a = Dropout(dropout_rate)(pooled_conv_a)

    # Branch b
    conv_b = Conv1D(filters=num_filters,
                    kernel_size=base_kernel_size + 1,
                    activation='relu',
                    )(embedding_dropped)
    pooled_conv_b = GlobalMaxPooling1D()(conv_b)
    pooled_conv_dropped_b = Dropout(dropout_rate)(pooled_conv_b)

    # Branch c
    conv_c = Conv1D(filters=num_filters,
                    kernel_size=base_kernel_size + 2,
                    activation='relu',
                    )(embedding_dropped)
    pooled_conv_c = GlobalMaxPooling1D()(conv_c)
    pooled_conv_dropped_c = Dropout(dropout_rate)(pooled_conv_c)

    # Collect branches into a single Convolution layer
    concat = Concatenate()(
        [pooled_conv_dropped_a, pooled_conv_dropped_b, pooled_conv_dropped_c])
    concat_dropped = Dropout(dropout_rate)(concat)

    # Dense output layer
    prob = Dense(units=1,  # dimensionality of the output space
                 activation='sigmoid',
                 )(concat_dropped)

    return Model(inputs, prob)


model = build_model(config.num_filters, config.dropout_rate,
                    config.base_kernel_size)
model.compile(loss=config.loss_function,
              optimizer=config.optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32,
          steps_per_epoch=len(x_train) / 32, epochs=config.epochs,
          validation_data=(x_val, y_val),
          callbacks=[WandbCallback(validation_data=(x_val, y_val),
                                   labels=["appropriate", "inappropriate"])])

# Will automatically log many metrics with the callback.
#wandb.log({'val_accuracy': accuracy})
