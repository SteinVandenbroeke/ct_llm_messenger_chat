import os
import json
from datetime import datetime
import torch
from torch.utils.data import Dataset


class Messenger_data(Dataset):
    def __init__(self, tokenizer, dataset_path="../datasets/messages_Niels", save_path="../datasets/messages.pt", max_length=512, context_window = 8):
        self.messages_data = {}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window

        if os.path.exists(save_path):
            self.messages_data = torch.load(save_path)
            print("Dataset loaded.")
        elif os.path.exists(dataset_path):
            print("Creating dataset..")
            self.messages_data = self.create_dataset(dataset_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.messages_data, save_path)
            print(f"Dataset created with {len(self.messages_data)} entries.")
        else:
            print("Dataset not found.")

    def create_dataset(self, messages_folder):
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
                        if len(participants) == 0:
                            continue

                        got_response = False
                        context = []
                        for msg in messages:
                            if "content" not in msg or "sender_name" not in msg:
                                continue
                            sender = msg["sender_name"]
                            content = msg["content"].replace("\n", "")
                            timestamp = datetime.fromtimestamp(msg["timestamp_ms"] / 1000)
                            formatted_time = timestamp.strftime("%A, %Y-%m-%d %I:%M %p")

                            if sender != user_name and got_response:
                                # Create sample with context and replies

                                # Put context in correct format
                                prompt = [{"role": "system", "content": f"You are a person called: {user_name}, You are chatting with: {participants}, " + (f"chat name: {chat_name}" if is_group else "")},]
                                context = context[-self.context_window:]
                                for m in context:
                                    prompt.append({"role": "assistant"  if m['sender'] == user_name else "user",
                                                        "content": f"[{m['sender']}][{m['time']}] {m['content']}"})

                                data[idx] = prompt

                                print(data[idx])
                                idx += 1

                                # Reset everything for next sample
                                context = []
                                got_response = False

                            # Add to context as prompt
                            if sender != user_name and not got_response:
                                context.append({
                                    "sender": sender,
                                    "content": content,
                                    "time": formatted_time,
                                })
                            # Add to result
                            else:
                                got_response = True
                                context.append({
                                    "sender": sender,
                                    "content": content,
                                    "time": formatted_time,
                                })
        return data

    def __getitem__(self, idx):
        messages = self.messages_data[idx]  # e.g. [{"role":"user","content":"Hello"}, ...]
        assistant_messages = [message for message in messages if message["role"] == "assistant"]

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

        #Labels are the items we want to generate -100 are masked values (we only want the assistant tokens in the labels tensor)
        labels = torch.full_like(tokens["input_ids"], fill_value=-100)

        for assistant_message in assistant_messages:
            assistant_tokens = self.tokenizer(
                assistant_message["content"],
                max_length=self.max_length,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=False
            )
            full_input_ids = tokens["input_ids"][0]
            assistant_input_ids = assistant_tokens["input_ids"][0][1:]
            #Serach for assistant_input_ids in full_input_ids and set them to the assistant_input_ids values
            for i in range(len(full_input_ids) - len(assistant_input_ids) + 1):
                if torch.equal(full_input_ids[i:i + len(assistant_input_ids)], assistant_input_ids):
                    labels[0, i:i + len(assistant_input_ids)] = full_input_ids[i:i + len(assistant_input_ids)]


        # tokens is a dict of tensors with batch dim 1, so squeeze it
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": labels[0]
        }

    def get_test_item(self, idx):
        messages = self.messages_data[idx]  # e.g. [{"role":"user","content":"Hello"}, ...]
        assistant_messages = [message for message in messages if message["role"] == "assistant"]

        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length=self.max_length,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length"
        )

        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "labels": tokens[0]
        }

    def __len__(self):
        return len(self.messages_data)