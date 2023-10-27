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

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Replace with your actual optimizer configuration

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Bird Types",
    page_icon=":bird:",
)


st.markdown("<h1 style='text-align: center;'>Bird Type Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>✨Welcome to my awesome DL task where I make a CNN network to classify types of birds! :)🐦</p>", unsafe_allow_html=True)

# Load the saved model
model_checkpoint_path = './model_checkpoint_v2.h5'  # Replace with your checkpoint path
model_new = tf.keras.models.load_model(model_checkpoint_path, custom_objects={'Adam': custom_optimizer})

# Constants
NUM_CLASSES = 10
IMG_SIZE = 64
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2
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
# Create the training dataset from the 'train' directory
    
train_ds = image_dataset_from_directory(
    directory='dataset/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

    # Create the validation dataset from the 'train' directory
validation_ds = image_dataset_from_directory(
    directory='dataset/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Define a function to build and train the model
def build_and_train_model(num_epochs, learning_rate, use_dropout):
    # Create a callback to save the best model during training
    model_checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Compile and train your model as usual
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_new.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if use_dropout:
        model_new.add(Dropout(0.5))  # You can adjust the dropout rate as needed
    
    history2 = model_new.fit(train_ds,
                            validation_data=validation_ds,
                            epochs=num_epochs,
                            callbacks=[model_checkpoint_callback],  # Add the callback here
                            verbose=1  # Set verbosity as desired
                            )

    return history2

training_complete = False

# Add a button to start training
if st.sidebar.button('Start Training'):
    st.text('Training in progress...')
    history = build_and_train_model(num_epochs, learning_rate, use_dropout)
    st.text('Training completed.')

    # Plot the loss and accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.tight_layout()
    st.pyplot(fig)
    training_complete = True

# Model Evaluation Section
st.header('Model Evaluation')

if training_complete:  # Check if the model is defined in the local scope
    # Load and display the confusion matrix
    # Assuming you've trained your 'model_new' using the training and validation datasets
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
else:
    st.write('Train the model to see evaluation results.')
