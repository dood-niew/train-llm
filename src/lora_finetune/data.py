"""Data collators for fine-tuning language models."""

from dataclasses import dataclass
from torch.utils.data import IterableDataset
import random
from transformers import PreTrainedTokenizer
from typing import Dict
from datasets import load_from_disk
from .arguments import DataArguments,TrainingArguments
import traceback
import torch
from .template import get_template 
from transformers import DataCollatorForSeq2Seq
IGNORE_INDEX=-100
def get_fitting_sequence(messages, tokenizer, max_length,model_template):
    for i in range(0,len(messages)-1,2):
        prompt = tokenizer.apply_chat_template(
            messages[i:-1], 
            tokenize=False,
            add_generation_prompt=True
        )

        assistant_response = messages[-1]['content']
        template_func = get_template(model_template)  
        full_text = template_func(prompt, assistant_response, tokenizer)

        encoded = tokenizer(
            full_text,
            max_length=max_length,
            truncation=False,
            padding=False,
            return_tensors="pt"
        )

        if len(encoded.input_ids[0]) <= max_length:
            return len(encoded.input_ids[0]), i

    return len(encoded.input_ids[0]), len(messages) - 1


class LoraDataset(IterableDataset):
    def __init__(self, datasets, seed, tokenizer, max_length,model_template,have_system):
        self._seed = seed
        self._datasets = datasets
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_len = len(datasets)
        self.model_template = model_template
        self.have_system = have_system
    def __iter__(self):
        return DatasetIterator(self._datasets, self._seed, self.tokenizer, self.max_length,self.model_template,self.have_system)

    def __len__(self):
        return self.total_len


class DatasetIterator:
    def __init__(self, datasets, seed, tokenizer, max_length,model_template,have_system=True):
        self._datasets = datasets
        self._rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_template = model_template
        self.have_system = have_system
    def __next__(self):
        while True:
            try:
                (dataset,) = self._rng.choices(self._datasets, k=1)

                messages = eval(dataset["text"])["messages"]
                
                system_prompt = messages[0]
                messages = messages[1:]
                    
                num_of_end_code, start_index = get_fitting_sequence(messages, self.tokenizer, self.max_length, self.model_template)
                if start_index == len(messages) - 1:
                    continue
                
                assistant_response = messages[-1]['content']
                
                if self.have_system:
                    n_of_system_promt=len(self.tokenizer.apply_chat_template([system_prompt]))
                    if n_of_system_promt+num_of_end_code < self.max_length:
                        messages = messages[start_index:]
                        messages.insert(0, system_prompt)
                    else:
                        messages = messages[start_index:]
                else:
                    n_of_system_promt=len(self.tokenizer(system_prompt['content']))
                    if n_of_system_promt+num_of_end_code+2 < self.max_length:
                        messages[-2]["content"] =  system_prompt['content']+"\n\n"+messages[-2]["content"]

                prompt = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True
                )

                # print("Prompt:", prompt)
                    
                template_func = get_template(self.model_template)
                full_text = template_func(prompt, assistant_response, self.tokenizer)

                encoded = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt",
                    add_special_tokens=False
                )

                print("Input:", self.tokenizer.decode(encoded.input_ids[0]))

                input_ids = encoded.input_ids[0]
                if input_ids.numel() == 0 or input_ids.size(0) == 0:
                    print("Empty input_ids. Skipping this item and continuing.")
                    print("Message:", messages)
                    continue
                attention_mask = encoded.attention_mask[0]
                labels = input_ids.clone()
                prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                labels[:prompt_length] = IGNORE_INDEX
                
                print("input_ids size",input_ids.size())
                print("labels size",labels.size())
                print("attention_mask size",attention_mask.size())

                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

            except StopIteration:
                print("Reached end of a dataset")
                raise

            except Exception as e:
                print(f"Error in next: {str(e)}. Skipping this item and continuing.")
                print("Message:", messages)
                traceback.print_exc()
                continue


def make_supervised_data_module(
    tokenizer:PreTrainedTokenizer, data_args: DataArguments, training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset = load_from_disk(data_args.data_train_path)
    train_dataset = LoraDataset(dataset,training_args.seed, tokenizer, training_args.model_max_length,data_args.model_template,data_args.have_system)
    eval_dataset = None
    if data_args.data_eval_path:
        eval_dataset = LoraDataset(load_from_disk(data_args.data_eval_path),training_args.seed, tokenizer, training_args.model_max_length,data_args.model_template,data_args.have_system)
    
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer,
    #     pad_to_multiple_of=8 if training_args.bf16 else None,
    #     label_pad_token_id=IGNORE_INDEX,
    #     padding='longest'
    # )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )