import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from data_processing import Messenger_data

class Messenger_fine_tuner:
    def __init__(self, model_id, dataset_path, output_dir):
        """
        Init function for Messenger_fine_tuner class
        :param model_id: name of the model being trained
        :param dataset_path: path to the dataset
        :param output_dir: directory to save the trained model
        """
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = Messenger_data(self.tokenizer, dataset_path=self.dataset_path, save_path=self.output_dir + "_messages.pt")
        print("Dataset Size: ", len(self.dataset))

    def prepare_model(self):
        """
        Model setup
        :return: model
        """
        # config for using smaller datatypes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # uses AutoModelForCausalLM for model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        )
        print("model prepared")
        return get_peft_model(model, peft_config)

    def train(self, epochs=0.1, batch_size=50, workers=16):
        """
        Trains the model
        :param epochs: number of epochs
        :param batch_size: size of the batch
        :param workers: amount of workers
        """
        logging.basicConfig(
            filename="training.log",
            filemode="w",             # "w" to overwrite each time, "a" to append
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO
        )
        print("Cuda available: ",torch.cuda.is_available())
        print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

        model = self.prepare_model()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            dataloader_num_workers=workers,
            learning_rate=2e-7,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            logging_dir="./logs",
            report_to=["tensorboard"],
            save_steps=100
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
        )


        trainer.train()
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)