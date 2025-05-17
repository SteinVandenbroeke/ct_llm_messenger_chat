import os

import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from collections import defaultdict


class Messenger_data(Dataset):
    def __init__(self, messages_folder="datasets/messages", save_path="datasets/messages.pt"):
        self.messages_data = {}
        if os.path.exists(save_path):
            self.messages_data = torch.load(save_path)
        else:
            self.messages_data = self.create_dataset(messages_folder)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.messages_data, save_path)

    def create_dataset(self, messages_folder):
        data = {}
        idx = 0

        for root, dirs, files in os.walk(messages_folder):
            dirs.sort()
            files.sort()
            for file in reversed(files):
                if file.startswith("message_") and file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf8") as f:
                        chat = json.load(f)

                        chat_name = chat["title"]
                        for message in reversed(chat["messages"]):
                            if "content" in message:
                                sender = message["sender_name"]
                                content = message["content"]
                                timestamp = datetime.fromtimestamp(message["timestamp_ms"] / 1000)
                                formatted_time = timestamp.strftime("%A %I:%M %p")

                                data[idx] = {
                                    "content": content,
                                    "sender": sender,
                                    "time": formatted_time,
                                    "chat_name": chat_name
                                }
                                idx += 1
        return data

    def __getitem__(self, index):
        return self.messages_data[index]

    def __len__(self):
        return len(self.messages_data)