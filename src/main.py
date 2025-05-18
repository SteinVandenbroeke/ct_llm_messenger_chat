from fine_tuning import Messenger_fine_tuner
from chatbot import MessengerChatbot

def main():
    # tuner = Messenger_fine_tuner(
    #     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     dataset_path="../datasets/messages_Niels",
    #     output_dir="../models/test2",
    # )
    # tuner.train()


    tester = MessengerChatbot(model_path="../model/test2")

    # 1. Test samples
    tester.test_model("../datasets/messages_Niels")

    # 2. Test custom prompt
    prompt = """
[Chat: Stein Vandenbroeke]
[Saturday 03:28 PM] [Stein Vandenbroeke] Hoelaat campus?
"""
    response = tester.generate_reply(prompt)
    print(f"Prompt: \n{prompt}")
    print(f"Model Reply: \n{response}")

# Prompt:
# <|user|>
# [Chat: Kachauw Kachauwð]
# [Sunday 01:42 PM] [Henri Kerch] Hebt gy die video gezien van "read"
#
#
# Actual response:

# [Sunday 01:42 PM] read?
#
# AI Response:
# [Sunday 01:42 PM] Nah nah
# [Sunday 01:42 PM] Dus ni
# [Sunday 01:42 PM] Er is nooit ni een video te zien van "read"
# [Sunday 01:42 PM] Of da ik zou zien

if __name__ == "__main__":
    main()