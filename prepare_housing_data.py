''' Helper functions to obtain housing data for Chapter Two of Geron'''
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(get_url=HOUSING_URL, get_path=HOUSING_PATH):
    ''' Get raw housing data from github'''
    if not os.path.isdir(get_path):
        os.makedirs(get_path)
    tgz_path = os.path.join(get_path, "housing.tgz")
    urllib.request.urlretrieve(get_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=get_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    ''' convert csv to df '''
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def get_and_load():
    ''' calls the above two functions in sequence '''
    fetch_housing_data()
    housing_df = load_housing_data()
    return housing_df


def split_data(data, test_ratio):
    ''' Split a dataframe into sets based on input ratio '''
    random_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = random_indicies[:test_set_size]
    train_indicies = random_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]
