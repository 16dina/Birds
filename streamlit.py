import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomTranslation, RandomZoom
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import image_dataset_from_directory

# Define a custom optimizer class
class CustomAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="CustomAdam", **kwargs):
        super(CustomAdam, self).__init__(name=name, **kwargs)

        self.learning_rate = learning_rate

    def get_config(self):
        config = super(CustomAdam, self).get_config()
        config.update({'learning_rate': self.learning_rate})
        return config

    def _create_slots(self, var_list):
        # Implement slot creation logic here, if needed
        pass

    def _resource_apply(self, grad, var, indices=None):
        # Implement resource apply logic here
        new_var = var - self.learning_rate * grad
        return tf.raw_ops.AssignVariableOp(ref=var, value=new_var)

custom_optimizer = CustomAdam(learning_rate=0.001)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Bird Types",
    page_icon=":bird:",
)

# Load the saved model
model_checkpoint_path = './model_checkpoint_v2.h5'  # Replace with your checkpoint path
model_new = tf.keras.models.load_model(model_checkpoint_path, custom_objects={'CustomAdam': CustomAdam})

# Constants
NUM_CLASSES = 10
IMG_SIZE = 64
batch_size = 32
image_size = (64, 64)
validation_split = 0.2

# EDA Section
st.header('Exploratory Data Analysis')

# Set the path to your dataset root directory
dataset_dir = './bird_images'

# List the subfolders (classes) in the dataset directory
classes = os.listdir(dataset_dir)

# Initialize a dictionary to store the image counts for each class
image_counts = {cls: 0 for cls in classes}

# Initialize a dictionary to store a few sample image paths from each class
sample_images = {cls: [] for cls in classes}

# Define the number of sample images to visualize
num_samples_to_visualize = 5

# Loop through each class and perform EDA
for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    class_images = os.listdir(class_dir)
    
    # Count the number of images in each class
    image_counts[cls] = len(class_images)
    
    # Randomly select a few sample images
    sample_images[cls] = random.sample(class_images, num_samples_to_visualize)

# Print the image counts for each class
for cls, count in image_counts.items():
    st.write(f"Class: {cls}, Number of Images: {count}")

# Visualize a few sample images from each class
for cls, images in sample_images.items():
    st.subheader(f"Sample Images from Class: {cls}")
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, image_name in enumerate(images):
        image_path = os.path.join(dataset_dir, cls, image_name)
        img = Image.open(image_path)
        with col1:
            st.image(img, caption=f'Image {i + 1}', use_column_width=True)
        col1, col2, col3, col4, col5 = col2, col3, col4, col5, st.empty()

# Calculate the class distribution (number of images per class)
class_counts = {cls: 0 for cls in classes}

# Loop through each class and count the number of images
for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    class_images = os.listdir(class_dir)
    class_counts[cls] = len(class_images)

# Create a bar plot to visualize the class distribution
st.subheader('Class Distribution')
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
st.pyplot(plt)  # Display the plot in Streamlit

# Model Training Section
st.header('Model Training')

# Create a sidebar for user inputs
st.sidebar.title('Training Parameters')
num_epochs = st.sidebar.slider('Number of Epochs', min_value=1, max_value=50, value=10)
learning_rate = st.sidebar.number_input('Learning Rate', min_value=0.001, max_value=1.0, value=0.001, step=0.001, format="%.3f")
use_dropout = st.sidebar.checkbox('Use Dropout')

# Assuming you've trained your 'model_new' using the training and validation datasets

# Model Evaluation Section
st.header('Model Evaluation')

# Load and display the confusion matrix
true_labels = []
predicted_labels = []

# Iterate through the validation dataset and make predictions
for batch in validation_ds:
    images, labels = batch
    true_labels.extend(np.argmax(labels, axis=1))  # Convert one-hot encoded labels to integers
    predictions = model_new.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

fig, ax = plt.subplots()

# Display the confusion matrix using the Streamlit `st.pyplot()` function
labels = list(range(NUM_CLASSES))  # Assuming your classes are numbered from 0 to NUM_CLASSES-1
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
display.plot(cmap='viridis', values_format='d', ax=ax)
st.pyplot(fig)
