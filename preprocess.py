from os import makedirs
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "./data"
makedirs(TRAIN_PATH := "./train", exist_ok=True)
makedirs(VALID_PATH := "./valid", exist_ok=True)
makedirs(TEST_PATH := "./test", exist_ok=True)

langs = ("eng", "deu", "esp")

data = {}

for lang in langs:
    train = pd.read_csv(f"{DATA_PATH}/train_{lang}.csv")
    valid = pd.read_csv(f"{DATA_PATH}/valid_{lang}.csv")
    test = pd.read_csv(f"{DATA_PATH}/test_{lang}.csv")
    data[lang] = pd.concat([train, valid, test], ignore_index=True, copy=False)
    data[lang].drop(columns="id", inplace=True)


for name, dataset in data.items():

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=7)
    valid_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=True, random_state=7)

    train_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)
    valid_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)
    test_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)

    train_data.to_csv(f"{TRAIN_PATH}/{name}.csv", index=False)
    valid_data.to_csv(f"{VALID_PATH}/{name}.csv", index=False)
    test_data.to_csv(f"{TEST_PATH}/{name}.csv", index=False)
