import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from collections import defaultdict


class Messenger_data(Dataset):
    def __init__(self, messages_folder="messages"):
        print(self.__len__())

    # TODO
    def __getitem__(self, index):
        return self.data[index], self.labels[index][1]

    # TODO
    def __len__(self):
        return self.n_samples