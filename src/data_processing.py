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
        info_file = messages_folder+"/autofill_information.json"
        if os.path.exists(info_file):
            with open(info_file, "r", encoding="utf8") as f:
                info = json.load(f)
                user_name = info["FULL_NAME"]
        else:
            # Default to name
            user_name = "Niels Van den Broeck" # Change


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

                        got_response = False
                        context = []
                        responses = []
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
                                prompt_parts = []
                                context = context[-self.context_window:]
                                for m in context:
                                    prompt_parts.append(f"[{m['time']}] [{m['sender']}] {m['content']}")
                                prompt = "[Chat: "+ chat_name + "]\n"+"\n".join(prompt_parts)

                                reply_parts = []
                                responses = responses[:self.context_window]
                                for m in responses:
                                    reply_parts.append(f"[{m['time']}] {m['content']}")
                                reply = "\n".join(reply_parts)

                                data[idx] = {
                                    "prompt": prompt,
                                    "responses": reply
                                }
                                print(data[idx])
                                idx += 1

                                # Reset everything for next sample
                                context = []
                                responses = []
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
                                responses.append({
                                    "content": content,
                                    "time": formatted_time,
                                })
        return data


    def __getitem__(self, idx):
        entry = self.messages_data[idx]

        full_text = f"<|user|>\n{entry['prompt']}\n<|assistant|>\n{entry['responses']}"

        tokens = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Just for checking
        if self.tokenizer(full_text, return_tensors="pt")["input_ids"].size(1) > self.max_length:
            print(f"Truncation occurred at index {idx}")

        tokens["labels"] = tokens["input_ids"].clone()  # Causal LM training
        return {k: v.squeeze() for k, v in tokens.items()}

    def __len__(self):
        return len(self.messages_data)