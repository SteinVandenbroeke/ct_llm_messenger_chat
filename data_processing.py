import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from collections import defaultdict


class Messenger_data(Dataset):
    def __init__(self, messages_folder="messages", save_path="data/messages.pt"):
        self.messages_data = {}
        print(self.__len__())
        if os.path.exists(save_path):
            self.messages_data = torch.load(save_path)
        else:
            self.messages_data = self.create_dataset(messages_folder)
            torch.save(self.messages_data, save_path)


    def create_dataset(self, messages_folder="messages"):


    # TODO
    def __getitem__(self, index):
        return self.messages_data[index]["message"], self.messages_data[index]["person"], self.messages_data[index]["time"], self.messages_data[index]["group"]

    # TODO
    def __len__(self):
        return self.messages_data.__len__()