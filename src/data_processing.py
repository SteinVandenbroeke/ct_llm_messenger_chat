import os
import json
from datetime import datetime
import torch
from torch.utils.data import Dataset


class Messenger_data(Dataset):
    def __init__(self, tokenizer, dataset_path="../datasets/messages", save_path="../datasets/messages.pt", max_length=5000, context_window = 15):
        """
        init fucnction for Messenger_data class
        :param tokenizer: used tokenizer, depends on model
        :param dataset_path: path to dataset to process/sample
        :param save_path: path to save processed dataset to
        :param max_length: max sample length in tokens
        :param context_window: max messages per sample
        """
        self.messages_data = {}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window

        # load dataset if already exists
        if os.path.exists(save_path):
            self.messages_data = torch.load(save_path)
            print("Dataset loaded.")
        # Create dataset based om path that is given
        elif os.path.exists(dataset_path):
            print("Creating dataset..")
            self.messages_data = self.create_dataset(dataset_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.messages_data, save_path)
            print(f"Dataset created with {len(self.messages_data)} entries.")
        else:
            print("Dataset not found.")

    def create_dataset(self, messages_folder):
        """
        Generates a pt file of dataset containing samples of messages
        :param messages_folder: path to the messenger data folder
        :return: dictionary of samples
        """
        data = {}
        idx = 0

        user_name = ""
        # Get name of user to train
        info_file = messages_folder + "/autofill_information.json"
        if os.path.exists(info_file):
            with open(info_file, "r", encoding="utf8") as f:
                info = json.load(f)
                user_name = info["autofill_information_v2"]["FULL_NAME"][0]
        else:
            # Default to name
            user_name = "Niels Van den Broeck" # Change

        print("User name:", user_name)

        for root, dirs, files in os.walk(messages_folder):
            # From old to new _9 -> _0
            dirs.sort()
            files.sort()
            for file in reversed(files):
                if file.startswith("message_") and file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf8") as f:
                        chat = json.load(f)
                        chat_name = chat.get("title", "Unknown chat")
                        messages = list(reversed(chat.get("messages", [])))  # Old to new
                        participants = [item[ "name"] for item in chat.get("participants", [])]
                        is_group = len(participants) > 2
                        # skip file is no participants
                        if len(participants) == 0:
                            continue
                        # Remove yourself from participants
                        if user_name in participants:
                            participants.remove(user_name)
                        participants =  ", ".join(participants)

                        context = []
                        for msg in messages:
                            # Skip message if no content is present or sender name is missing
                            if "content" not in msg or "sender_name" not in msg:
                                continue
                            sender = msg["sender_name"]
                            content = msg["content"].replace("\n", "")
                            timestamp = datetime.fromtimestamp(msg["timestamp_ms"] / 1000)
                            # Convert time to readable format for model
                            formatted_time = timestamp.strftime("%A, %Y-%m-%d %I:%M %p")

                            # Append message to context
                            context.append({
                                "sender": sender,
                                "content": content,
                                "time": formatted_time,
                            })

                            # End sample if last message is you and context window is reached
                            if msg["sender_name"] == user_name and len(context) >= self.context_window:
                                # Put context in correct format
                                prompt = [{"role": "system", "content": f"You are a person called: {user_name}. You are chatting with: {participants}. " + (f"Chat name: {chat_name}." if is_group else "") + f" Its importand to answer in the chat format '[sender][time] message' where sender is always {user_name}, because you rollplaying this person."}]
                                # Adjust to context window
                                context = context[-self.context_window:]
                                for m in context:
                                    prompt.append({"role": "assistant"  if m['sender'] == user_name else "user",
                                                        "content": f"[{m['sender']}][{m['time']}] {m['content']}"})
                                # Save sample in data dict
                                data[idx] = prompt

                                idx += 1

                                # Reset everything for next sample
                                context = []

        return data

    def __getitem__(self, idx):
        """
        returns the sample at index idx
        :param idx: index
        :return: sample
        """

        # get sample from saved dataset
        messages = self.messages_data[idx]  # e.g. [{"role":"user","content":"Hello"}, ...]

        # Tokenize messages in chat template
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=self.max_length,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length"
        )

        # Just for checking
        if "overflowing_tokens" in tokens and tokens["overflowing_tokens"].size(1) > 0:
            print("Truncation occurred.")

        # Tokens is a dict of tensors with batch dim 1, so squeeze it
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels":  tokens["input_ids"][0].clone()
        }

    def get_test_item(self, idx):
        """
        returns a sample without corresponding response (from you). This is used to evaluate the model without leaking.
        :param idx: index
        :return: sample
        """
        messages = self.messages_data[idx]  # e.g. [{"role":"user","content":"Hello"}, ...]

        # Find index of last not you user, after that is response
        start_index = 0
        for i in reversed(range(len(messages))):
            if messages[i]["role"] == "user":
                start_index = i

        tokens = self.tokenizer.apply_chat_template(
            messages[:start_index], # Cut everything after
            tokenize=True,
            add_generation_prompt=True,
            max_length=self.max_length,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
            padding=False
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels":  tokens["input_ids"].clone()
        }

    def __len__(self):
        return len(self.messages_data)