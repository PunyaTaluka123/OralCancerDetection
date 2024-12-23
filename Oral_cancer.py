import tensorflow as tf

# Load the uncompiled model
model = tf.keras.models.load_model('keras_model.h5', compile=False)

# Compile the model with the desired settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # Use 'binary_crossentropy' for binary classification
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save the compiled model if needed
model.save('compiled_model.h5')
