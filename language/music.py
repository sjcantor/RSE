"""Code for loading the parsed MusicNet dataset"""

from nis import match
import os
from re import sub
from unittest import case
import numpy as np
from subprocess import run

import config as cnf
import data_utils
from language.utils import LanguageTask

# musicnet_data_path = os.path.join(cnf.repo_dir, 'musicnet_data')
# maestro_data_path = os.path.join(cnf.repo_dir, 'maestro_data')
# data_path = maestro_data_path if cnf.dataset == 'maestro' else musicnet_data_path
# print(f'path for dataset:{data_path}')
str_fourier = f"fourier{cnf.musicnet_fourier_multiplier}" if cnf.musicnet_do_fourier_transform else "raw"

# MUSICNET_TRAIN = os.path.join(musicnet_data_path, f"musicnet_{str_fourier}_train_{cnf.musicnet_file_window_size}.npy")
# MUSICNET_VALIDATION = os.path.join(musicnet_data_path, f"musicnet_{str_fourier}_validation_{cnf.musicnet_file_window_size}.npy")
# MUSICNET_TEST = os.path.join(musicnet_data_path, f"musicnet_{str_fourier}_test_{cnf.musicnet_file_window_size}.npy")

# MAESTRO_TRAIN = os.path.join(maestro_data_path, f"maestro_{str_fourier}_train_{cnf.musicnet_file_window_size}.npy")
# MAESTRO_VALIDATION = os.path.join(maestro_data_path, f"maestro_{str_fourier}_validation_{cnf.musicnet_file_window_size}.npy")
# MAESTRO_TEST = os.path.join(maestro_data_path, f"maestro_{str_fourier}_test_{cnf.musicnet_file_window_size}.npy")

data_folder = os.path.join(cnf.repo_dir, f"{cnf.dataset}_data")
TRAIN = os.path.join(data_folder, f"{cnf.dataset}_{str_fourier}_train_{cnf.musicnet_file_window_size}.npy")
VALIDATION = os.path.join(data_folder, f"{cnf.dataset}_{str_fourier}_validation_{cnf.musicnet_file_window_size}.npy")
TEST = os.path.join(data_folder, f"{cnf.dataset}_{str_fourier}_test_{cnf.musicnet_file_window_size}.npy")

# print(f'[Debug log] Data folder: {data_folder} \nData train file: {TRAIN}')

# TRAIN = MAESTRO_TRAIN if cnf.dataset == 'maestro' else MUSICNET_TRAIN
# VALIDATION = MAESTRO_VALIDATION if cnf.dataset == 'maestro' else MUSICNET_VALIDATION
# TEST = MAESTRO_TEST if cnf.dataset == 'maestro' else MUSICNET_TEST

def get_parsed_dataset():
    print(f"No training set found that matches config.py. Getting {cnf.dataset} and parsing it.")
    run([cnf.python_ver, os.path.join(cnf.repo_dir, 'get_dataset.py')])  # download musicnet if it is missing
    run([cnf.python_ver, os.path.join(cnf.repo_dir, 'parse_file.py')])  # parse file so it can be processed by the model


# Renamed this from 'Musicnet' so that it can be generalized
class MusicDataset(LanguageTask):
    def __init__(self) -> None:
        self.window_size = cnf.musicnet_window_size  # how long is one input sequence (e.g. 2048)
        self.mmap_count = cnf.musicnet_mmap_count  # how many inputs there are in mmap partial load
        self.training_set = []
        self.validation_set = []
        self.testing_set = []
        # TODO - shouldn't get dataset when transcribe creates this class
        # if not os.path.exists(MUSICNET_TRAIN):
        #     get_parsed_musicnet()
        if not os.path.exists(TRAIN):
            get_parsed_dataset()

    def crop(self, xy_set):
        """Crop data to a smaller sized window."""
        midpoint = cnf.music_file_window_size // 2
        half_window = self.window_size // 2
        result = xy_set[:, :, midpoint - half_window:midpoint + half_window]
        return list(result)  # returns [[[inputs1],[labels1]],..] (,2,window_size)

    def load_training_dataset(self):
        loaded = np.load(TRAIN)
        self.training_set = self.crop(loaded)
        del loaded

    def sample_training_dataset_mmap(self):
        n_sample_locations = self.mmap_count
        loaded = np.load(TRAIN, mmap_mode='r')
        mmap_window = self.mmap_count // n_sample_locations
        indices = np.random.randint(low=0, high=len(loaded) - mmap_window, size=n_sample_locations)
        training_set_tmp = []
        for i in range(n_sample_locations):
            training_set_tmp += list(loaded[indices[i]:indices[i] + mmap_window])
        self.training_set = self.crop(np.array(training_set_tmp, dtype='float32'))
        del loaded

    def prepare_data(self):
        print(f"Loading the {cnf.dataset} dataset", flush=True)
        self.prepare_train_data()
        self.prepare_validation_data()
        data_utils.reset_counters()

    def prepare_train_data(self):
        if cnf.music_subset:
            self.sample_training_dataset_mmap()
        else:
            self.load_training_dataset()
        # print(f'DEBUG: data_utils keys: {data_utils.train_set.keys()}')
        data_utils.train_set[f"{cnf.task}"][self.window_size] = self.training_set

    def prepare_validation_data(self):
        loaded = np.load(VALIDATION)
        self.validation_set = self.crop(loaded)
        del loaded
        data_utils.test_set[f"{cnf.task}"][self.window_size] = self.validation_set

    def prepare_test_data(self):
        loaded = np.load(TEST)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set[f"{cnf.task}"][self.window_size] = self.testing_set

    def prepare_inference_data(self, inference_file_path):
        loaded = np.load(inference_file_path)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set[f"{cnf.task}"][self.window_size] = self.testing_set
