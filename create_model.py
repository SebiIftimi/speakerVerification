"""
This script is used to preprocess audio files, define, train, and save a Siamese neural network model 
for speaker verification.

Workflow:
1. Load Data:
   - Uses `load_siamese_data` to load pairs of audio data from JSON.
   - Stores negative pairs in `X1`, positive pairs in `X2`, and labels in `y`.
   - Specify JSON file path in `DATA_PATH`.

2. Data Preprocessing:
   - Extracts MFCC features for each audio file.
   - Standardizes and normalizes the MFCCs for consistent, compact representations.

Model Design:
   - A Siamese Neural Network learns to distinguish between speakers.
   - Consists of identical subnetworks for feature extraction, sharing parameters and weights.
   - Computes feature similarity between paired inputs to determine if they are from the same speaker.

Python and Library Requirements:
   - Python Version: 3.9.6
   - TensorFlow: 2.16.2, Librosa: 0.10.2, Scikit-learn: 1.5.1, Matplotlib: 3.9.2
   - Using these versions is recommended for compatibility.

"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from tensorflow.keras import layers, Model

DATA_PATH = "siamese_dataset.json"
SAMPLE_RATE = 16000
NUM_MFCC = 13
MAX_LEN = 1300  # Maximum length for MFCC features

def load_siamese_data(data_path):
    """Loads and preprocesses the Siamese pairs dataset from JSON file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    pairs = data["pairs"]
    labels = np.array(data["labels"])

    X1 = []
    X2 = []
    y = []

    for idx, pair in enumerate(pairs):
        # Check if both files in the pair are .wav files
        if pair[0].endswith(('.wav')) and pair[1].endswith(('.wav')):
            try:
                # Extract MFCC features for both files in the pair
                mfcc_1 = preprocess_audio(pair[0])
                mfcc_2 = preprocess_audio(pair[1])
                X1.append(mfcc_1)
                X2.append(mfcc_2)
                y.append(labels[idx])
            except Exception as e:
                print(f"Error processing pair {pair}: {e}")

    # Convert lists to numpy arrays for model compatibility
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)

    return X1, X2, y

def preprocess_audio(file_path):
    """Extracts MFCC features from an audio file and applies padding and normalization."""
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=NUM_MFCC)
    mfcc = mfcc.T

    # Pad or truncate MFCC to a consistent length
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

    # Normalize MFCC features
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-10)

    return mfcc 

def prepare_datasets(test_size, validation_size):
    """Splits dataset into training, validation, and test sets."""
    X1, X2, y = load_siamese_data(DATA_PATH)

    # Initial train-test split
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=test_size, random_state=42)

    # Split training set further to obtain a validation set
    X1_train, X1_validation, X2_train, X2_validation, y_train, y_validation = train_test_split(
        X1_train, X2_train, y_train, test_size=validation_size, random_state=42)
    
    return (X1_train, X2_train, y_train), (X1_validation, X2_validation, y_validation), (X1_test, X2_test, y_test)

def plot_history(history):
    """Plots training and validation accuracy and loss over epochs."""
    fig, axs = plt.subplots(2)

    # Plot accuracy
    axs[0].plot(history.history["accuracy"], label="train_accuracy")
    axs[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    # Plot loss
    axs[1].plot(history.history["loss"], label="train_loss")
    axs[1].plot(history.history["val_loss"], label="val_loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss Evaluation")

    plt.show()

class L1DistanceLayer(layers.Layer):
    """Defines L1 distance as a custom layer for the Siamese network."""
    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)

def build_base_network_lstm(input_shape):
    """Defines the shared LSTM-based subnetwork for the Siamese network."""
    model = keras.Sequential()
    
    # Add LSTM layers for sequential processing of audio features
    model.add(layers.LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(layers.BatchNormalization())
    
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.BatchNormalization())
    
    model.add(layers.LSTM(64))
    model.add(layers.BatchNormalization())

    # Dense layers with dropout for regularization
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.2))

    return model

def build_siamese_model_lstm(input_shape):
    """
    Constructs the Siamese model architecture.
    - Loads two audio files as inputs `input_a` and `input_b`.
    - Processes each input through the shared subnetwork.
    - Calculates the L1 distance between the two feature vectors.
    - The final model returns a similarity score: lower distance suggests same speaker; higher distance suggests different speakers.
    """
    base_network = build_base_network_lstm(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Process each input through the shared LSTM subnetwork
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Calculate L1 distance between the processed inputs
    l1_distance = L1DistanceLayer()([processed_a, processed_b])

    # Further dense layers for similarity score calculation
    distance = layers.Dense(128, activation="relu")(l1_distance)
    distance = layers.Dropout(0.3)(distance)
    
    distance = layers.Dense(128, activation="relu")(distance)
    distance = layers.Dropout(0.3)(distance)

    # Final output layer with sigmoid activation for binary classification (same/different speaker)
    output = layers.Dense(1, activation='sigmoid')(distance)

    siamese_model = Model([input_a, input_b], output)

    return siamese_model

if __name__ == "__main__":
    # Prepare datasets for training, validation, and testing
    (X1_train, X2_train, y_train), (X1_validation, X2_validation, y_validation), (X1_test, X2_test, y_test) = prepare_datasets(0.25, 0.2)

    input_shape = (MAX_LEN, NUM_MFCC)  # Define input shape based on MFCC dimensions

    # Build and compile the Siamese model
    siamese_model = build_siamese_model_lstm(input_shape)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Training callbacks for learning rate reduction and early stopping
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = siamese_model.fit([X1_train, X2_train], y_train, validation_data=([X1_validation, X2_validation], y_validation),
                                batch_size=32, epochs=50, callbacks=[lr_reduction, early_stopping])

    # Evaluate on the test set
    test_error, test_accuracy = siamese_model.evaluate([X1_test, X2_test], y_test, verbose=2)
    print(f"Accuracy on test set is {test_accuracy}")

    # Plot accuracy and loss history
    plot_history(history)

    # Save the trained model
    siamese_model.save("siamese_speaker_verification_model.h5")
