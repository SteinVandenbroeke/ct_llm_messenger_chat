import os

import torch
from transformers import pipeline

from fine_tuning import Messenger_fine_tuner
from chatbot import MessengerChatbot
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


def main():
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

    # Load model directly
    # #
    # tuner = Messenger_fine_tuner(
    #     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     dataset_path="../datasets/messages",
    #     output_dir="../models/test4",
    # )
    # tuner.train()
    #
    tuner = Messenger_fine_tuner(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_path="../datasets/messages",
        output_dir="../models/test5",
    )
    print('argument list', sys.argv)
    tuner.train(float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))

    tester = MessengerChatbot(model_path="../models/test5")

    # 1. Test samples
    tester.test_model("../datasets/messages")

    # 2. Test custom prompt
    prompt = """
[Chat: Stein Vandenbroeke]
[Saturday 03:28 PM] [Stein Vandenbroeke] Hoelaat campus?
"""
    response = tester.generate_reply(prompt)
    print(f"Prompt: \n{prompt}")
    print(f"Model Reply: \n{response}")



if __name__ == "__main__":
    main()