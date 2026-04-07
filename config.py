"""Central configuration file for all project constants."""

import os


# Dataset settings
MAX_SEQ_LENGTH = 512
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model settings
MODEL_NAME_DISTILBERT = "distilbert-base-uncased"
MODEL_NAME_BASELINE = "tfidf_logistic"

# Training settings
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_SAMPLES = 5000

# Path settings
DATA_CACHE_DIR = "./data/cache"
MODEL_SAVE_DIR = "./models"
RESULTS_DIR = "./results"


def create_directories():
    """Create necessary directories if they do not exist.
    
    Returns:
        True if all directories created successfully, False otherwise.
    """
    dirs_to_create = [
        DATA_CACHE_DIR,
        MODEL_SAVE_DIR,
        RESULTS_DIR
    ]
    
    for directory in dirs_to_create:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory {directory}: {error}")
            return False
    
    return True


if __name__ == "__main__":
    print("Creating project directories...")
    
    success = create_directories()
    
    if success:
        print("All directories created successfully")
        print(f"Data cache: {DATA_CACHE_DIR}")
        print(f"Models: {MODEL_SAVE_DIR}")
        print(f"Results: {RESULTS_DIR}")
    else:
        print("Failed to create some directories")