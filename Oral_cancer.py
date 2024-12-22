import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('keras_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image if necessary
    return img_array

# Function to generate Grad-CAM heatmap


def get_grad_cam_heatmap(model, img_array, target_layer_name, class_index=None):
    # Extract the target layer
    target_layer = model.get_layer("sequential_13").get_layer(
        "model4").get_layer(target_layer_name)

    # Create a model that outputs both the target layer and the model's predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    # Compute the gradients and class activation map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            # If no class_index is provided, take the predicted class
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    # Get the gradients of the output class with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(
        0, 1, 2))  # Pool gradients spatially

    # Get the convolutional output
    conv_outputs = conv_outputs[0]

    # Create the heatmap by multiplying the pooled gradients with the feature map
    # Weighted sum of feature maps
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)  # ReLU to remove negative values
    # Normalize between 0 and 1
    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Function to visualize the Grad-CAM heatmap


def visualize_grad_cam(img_path, heatmap):
    # Load the original image
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)

    # Resize heatmap to match the image size
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    heatmap = np.expand_dims(heatmap, axis=-1)  # Add channel dimension
    heatmap = np.tile(heatmap, (1, 1, 3))  # Convert grayscale to RGB
    heatmap = np.uint8(heatmap)
    # Resize heatmap to image size
    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1]))

    # Superimpose the heatmap on the image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        img * 0.7 + heatmap * 0.3)

    # Display the image and heatmap
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()


# Main part of the code
# Path to the input image
img_path = 'OralCancer\\cancer\\3-s2.0-B9780443100734500148-f10-07-9780443100734.jpg'
img_array = preprocess_image(img_path)

# Get the Grad-CAM heatmap
heatmap = get_grad_cam_heatmap(model, img_array, target_layer_name='Conv_1')

# Visualize the Grad-CAM heatmap
visualize_grad_cam(img_path, heatmap)
