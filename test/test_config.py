"""Test configuration constants and directory creation."""

import os
import shutil
import config


def test_constants_exist():
    """Verify all expected constants are defined."""
    assert hasattr(config, "MAX_SEQ_LENGTH")
    assert hasattr(config, "TEST_SIZE")
    assert hasattr(config, "RANDOM_STATE")
    assert hasattr(config, "MODEL_NAME_DISTILBERT")
    assert hasattr(config, "MODEL_NAME_BASELINE")
    assert hasattr(config, "BATCH_SIZE")
    assert hasattr(config, "NUM_EPOCHS")
    assert hasattr(config, "LEARNING_RATE")
    assert hasattr(config, "MAX_SAMPLES")
    assert hasattr(config, "DATA_CACHE_DIR")
    assert hasattr(config, "MODEL_SAVE_DIR")
    assert hasattr(config, "RESULTS_DIR")


def test_constant_types():
    """Verify constants have correct types."""
    assert isinstance(config.MAX_SEQ_LENGTH, int)
    assert isinstance(config.TEST_SIZE, float)
    assert isinstance(config.RANDOM_STATE, int)
    assert isinstance(config.MODEL_NAME_DISTILBERT, str)
    assert isinstance(config.BATCH_SIZE, int)
    assert isinstance(config.LEARNING_RATE, float)


def test_constant_values():
    """Verify constants have reasonable values."""
    assert config.MAX_SEQ_LENGTH == 512
    assert config.TEST_SIZE == 0.2
    assert config.BATCH_SIZE == 16
    assert config.NUM_EPOCHS == 3
    assert config.MAX_SAMPLES == 5000
    assert config.MODEL_NAME_DISTILBERT == "distilbert-base-uncased"


def test_create_directories():
    """Verify directory creation works."""
    success = config.create_directories()
    
    assert success is True
    assert os.path.exists(config.DATA_CACHE_DIR)
    assert os.path.exists(config.MODEL_SAVE_DIR)
    assert os.path.exists(config.RESULTS_DIR)


def test_directories_are_paths():
    """Verify path constants are strings."""
    assert isinstance(config.DATA_CACHE_DIR, str)
    assert isinstance(config.MODEL_SAVE_DIR, str)
    assert isinstance(config.RESULTS_DIR, str)
    
    assert config.DATA_CACHE_DIR.startswith("./")
    assert config.MODEL_SAVE_DIR.startswith("./")
    assert config.RESULTS_DIR.startswith("./")


def teardown_module():
    """Clean up test directories after tests."""
    dirs_to_clean = [
        config.DATA_CACHE_DIR,
        config.MODEL_SAVE_DIR,
        config.RESULTS_DIR
    ]
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except OSError:
                pass