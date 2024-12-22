import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('keras_model.h5')
sequential_16 = model.get_layer('sequential_13').get_layer(
    "model4")
print(sequential_16.summary())
