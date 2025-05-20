import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from data_processing import Messenger_data
import string

class MessengerChatbot:
    def __init__(self, model_path, max_length=3000):
        """
        Init function for MessengerChatbot Class
        :param model_path:
        :param max_length:
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.max_length = max_length


    def generate_reply(self, prompt, temperature=0.2):
        """
        generates tokens of a given prompt
        :param prompt: prompt in following format: [{"role": "user", "content": "Hello"}, ...]
        :param temperature: randomness
        :return: tokens of the prompt
        """

        inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=self.max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )



        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # Return only the generated part after <|assistant|>
        return generated




    def test_model(self, dataset_path, temperature=0.7):
        """
        takes 100 random samples. cuts the response and passes it to the model to check the result.
        :param dataset_path: path of the dataset
        :param temperature: randomness
        """
        dataset = Messenger_data(self.tokenizer, dataset_path=dataset_path)

        # 100 random samples
        random_numbers = random.sample(range(len(dataset)), 100)
        for num in random_numbers:
            #item = dataset.get_test_item(num)
            item = dataset[num]
            input_ids = item['input_ids']
            sample = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            print("-- Actual --")
            print(sample)
            print("------------")
            inputs = dataset.get_test_item(num)["input_ids"].to(self.device)

            print(len(inputs))
            # Generates response
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs,
                    max_length=self.max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"AI Response:\n{generated}\n")
            print("-------------------------------------------------------------")
            input()