"""Code for loading the parsed MusicNet dataset"""

import os
import numpy as np
from subprocess import run

import config as cnf
import data_utils
from language.utils import LanguageTask

musicnet_data_path = os.path.join(cnf.repo_dir, 'musicnet_data')
str_fourier = f"fourier{cnf.musicnet_fourier_multiplier}" if cnf.musicnet_do_fourier_transform else "raw"

MUSICNET_TRAIN = os.path.join(musicnet_data_path, f"musicnet_{str_fourier}_train_{cnf.musicnet_file_window_size}.npy")
MUSICNET_VALIDATION = os.path.join(musicnet_data_path, f"musicnet_{str_fourier}_validation_{cnf.musicnet_file_window_size}.npy")
MUSICNET_TEST = os.path.join(musicnet_data_path, f"musicnet_{str_fourier}_test_{cnf.musicnet_file_window_size}.npy")

def get_parsed_musicnet():
    print("No training set found that matches config.py. Getting musicnet and parsing it.")
    run([cnf.python_ver, os.path.join(musicnet_data_path, 'get_musicnet.py')])  # download musicnet if it is missing
    run([cnf.python_ver, os.path.join(musicnet_data_path, 'parse_file.py')])  # parse file so it can be processed by the model


class Musicnet(LanguageTask):
    def __init__(self) -> None:
        self.window_size = cnf.musicnet_window_size  # how long is one input sequence (e.g. 2048)
        self.mmap_count = cnf.musicnet_mmap_count  # how many inputs there are in mmap partial load
        self.training_set = []
        self.validation_set = []
        self.testing_set = []
        # TODO - shouldn't get dataset when transcribe creates this class
        # if not os.path.exists(MUSICNET_TRAIN):
        #     get_parsed_musicnet()

    def crop(self, xy_set):
        """Crop data to a smaller sized window."""
        midpoint = cnf.musicnet_file_window_size // 2
        half_window = self.window_size // 2
        result = xy_set[:, :, midpoint - half_window:midpoint + half_window]
        return list(result)  # returns [[[inputs1],[labels1]],..] (,2,window_size)

    def load_training_dataset(self):
        loaded = np.load(MUSICNET_TRAIN)
        self.training_set = self.crop(loaded)
        del loaded

    def sample_training_dataset_mmap(self):
        n_sample_locations = self.mmap_count
        loaded = np.load(MUSICNET_TRAIN, mmap_mode='r')
        mmap_window = self.mmap_count // n_sample_locations
        indices = np.random.randint(low=0, high=len(loaded) - mmap_window, size=n_sample_locations)
        training_set_tmp = []
        for i in range(n_sample_locations):
            training_set_tmp += list(loaded[indices[i]:indices[i] + mmap_window])
        self.training_set = self.crop(np.array(training_set_tmp, dtype='float32'))
        del loaded

    def prepare_data(self):
        print("Loading the dataset", flush=True)
        self.prepare_train_data()
        self.prepare_validation_data()
        data_utils.reset_counters()

    def prepare_train_data(self):
        if cnf.musicnet_subset:
            self.sample_training_dataset_mmap()
        else:
            self.load_training_dataset()
        data_utils.train_set["musicnet"][self.window_size] = self.training_set

    def prepare_validation_data(self):
        loaded = np.load(MUSICNET_VALIDATION)
        self.validation_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.validation_set

    def prepare_test_data(self):
        loaded = np.load(MUSICNET_TEST)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.testing_set

    def prepare_inference_data(self, inference_file_path):
        loaded = np.load(inference_file_path)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.testing_set
