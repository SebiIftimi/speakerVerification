"""
This script is used to perform speaker verification predictions using a pre-trained Siamese model.

Note:
    The audio files used for comparison must match the format of those used to train the model:
        - Format: Mono
        - Sample Rate: 16kHz
    Ensure consistency with the training setup:
        - SAMPLE_RATE = 16000
        - NUM_MFCC = 13
        - MAX_LEN = 1300 (length of MFCC features)
    MFCC feature extraction is performed using the 'librosa' library.

Requirements:
    - Python Version: 3.9.6
    - TensorFlow Version: 2.16.2
    - Librosa Version: 0.10.2
    It is recommended to use these specific versions for compatibility.

To use this script, specify the path to the pre-trained model in "MODEL_PATH".
    - file_path_1: Path to the first audio file.
    - file_path_2: Path to the second audio file.
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import librosa

class L1DistanceLayer(keras.layers.Layer):
    """Defines L1 Distance calculation for use in Siamese network."""
    def __init__(self, **kwargs):
        super(L1DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Computes the absolute difference between two inputs
        return tf.abs(inputs[0] - inputs[1])

MODEL_PATH = "/Users/sebiiftimi/Desktop/SuntDesteptFacIA/SpeakerVerification/siamese_speaker_identification_model.h5"

def preprocess_audio(file_path):
    """Preprocesses an audio file before inputting it to the model for prediction.
    
    Steps:
    1. Extract MFCC features using 'librosa'.
    2. Apply padding or truncation to ensure consistent dimensions for model input.
    3. Normalize MFCC features.
    """
    SAMPLE_RATE = 16000
    NUM_MFCC = 13
    MAX_LEN = 1300

    # Load audio file with specified sample rate
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=NUM_MFCC)
    mfcc = mfcc.T

    # Pad or truncate to match model's expected input shape
    if len(mfcc) > MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

    # Normalize MFCCs
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-10)

    return mfcc

def predict_if_same_speaker(model, file_path_1, file_path_2):
    """Applies the Siamese model to determine if two audio files are from the same speaker.
    
    Decision Threshold:
        A threshold of 0.6 is used to interpret the model’s output.
        - If prediction > 0.6: The two audio files are from the same speaker.
        - If prediction < 0.6: The two audio files are from different speakers.
    """
    # Preprocess both audio files to obtain MFCC features
    mfcc_1 = preprocess_audio(file_path_1)
    mfcc_2 = preprocess_audio(file_path_2)

    # Expand dimensions to fit model's expected input shape
    mfcc_1 = np.expand_dims(mfcc_1, axis=0)
    mfcc_2 = np.expand_dims(mfcc_2, axis=0)

    # Generate similarity prediction using the model
    prediction = model.predict([mfcc_1, mfcc_2])

    # Interpret the prediction based on the threshold
    if prediction > 0.6:
        return prediction, "Same speaker"
    else:
        return prediction, "Different speaker"

if __name__ == "__main__":
    # Load the trained Siamese model, specifying the custom L1DistanceLayer
    model = keras.models.load_model(MODEL_PATH, custom_objects={'L1DistanceLayer': L1DistanceLayer})

    # Specify paths to the audio files to compare
    file_path_1 = "path-to-first-audio-file"
    file_path_2 = "path-to-second-audio-file"
    
    # Run prediction
    probability, result = predict_if_same_speaker(model, file_path_1, file_path_2)

    # Display result
    print(f"Probability of same speaker in both audio files: {probability[0][0]:.4f}")
    print(result)
