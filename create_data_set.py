"""
This script generates a JSON file with data for training a speaker verification model.

Input:
 - "DATASET_PATH": The path to the directory containing subdirectories of audio files for each speaker.

Output:
 - "dataset.json": This file stores paths to pairs of audio files labeled as positive (same speaker) or negative (different speakers):
       - Positive pairs: e.g., voice1 - voice1, voice2 - voice2, voice3 - voice3, etc.
       - Negative pairs: e.g., voice1 - voice2, voice16 - voice200, voice3 - voice34, etc.

Note: The JSON file must be created in advance.

The directory specified in "DATASET_PATH" should contain a separate folder for each speaker, with multiple recordings per speaker.
The program creates:
 - Positive pairs by randomly selecting two recordings from the same folder.
 - Negative pairs by randomly selecting recordings from two different speaker folders.

The JSON file stores paths to each pair along with a label indicating whether they are positive or negative:
 - 0: Negative pair (different speakers)
 - 1: Positive pair (same speaker)

This approach avoids storing actual audio files in JSON, which would make the file excessively large and difficult to manage during model training. Instead, the model reads audio files from specified paths during training.

Usage:
 - Specify the following parameters:
     - DATASET_PATH: Path to the dataset directory containing speaker audio files.
     - OUTPUT_PATH: Path to the JSON file where positive/negative pairs and labels are saved.

Setting "NUM_PAIRS_PER_CLASS":
 - NUM_PAIRS_PER_CLASS: Defines how many pairs to generate. For example, if NUM_PAIRS_PER_CLASS = 500, the script generates 500 positive and 500 negative pairs.
"""

import os
import random
import json

DATASET_PATH = "/path/to/dataset_directory"
OUTPUT_PATH = "/path/to/output_json_file"
NUM_PAIRS_PER_CLASS = 500 

def create_siamese_pairs(dataset_path, num_pairs_per_class):
    """Generates positive and negative audio file pairs for training a Siamese network."""

    data = {
        "pairs": [],
        "labels": []
    }

    # Get list of speaker directories
    speakers = os.listdir(dataset_path)
    speakers = [speaker for speaker in speakers if os.path.isdir(os.path.join(dataset_path, speaker))]

    # Generate positive pairs (same speaker)
    for speaker in speakers:
        speaker_path = os.path.join(dataset_path, speaker)
        audio_files = os.listdir(speaker_path)
        audio_files = [os.path.join(speaker_path, file) for file in audio_files if file.endswith('.wav')]

        for _ in range(num_pairs_per_class):
            file1, file2 = random.sample(audio_files, 2)
            data["pairs"].append([file1, file2])
            data["labels"].append(1)  # Positive pair

    # Generate negative pairs (different speakers)
    for _ in range(num_pairs_per_class * len(speakers)):
        speaker1, speaker2 = random.sample(speakers, 2)
        speaker1_files = os.listdir(os.path.join(dataset_path, speaker1))
        speaker2_files = os.listdir(os.path.join(dataset_path, speaker2))

        file1 = os.path.join(dataset_path, speaker1, random.choice(speaker1_files))
        file2 = os.path.join(dataset_path, speaker2, random.choice(speaker2_files))

        data["pairs"].append([file1, file2])
        data["labels"].append(0)  # Negative pair

    return data

if __name__ == "__main__":
    siamese_data = create_siamese_pairs(DATASET_PATH, NUM_PAIRS_PER_CLASS)

    with open(OUTPUT_PATH, "w") as fp:
        json.dump(siamese_data, fp, indent=4)

    print(f"Dataset saved to {OUTPUT_PATH}.")
