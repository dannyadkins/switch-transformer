import torch 
from model import SwitchTransformer

import datasets 
from datasets import load_dataset
from typing import Literal, Callable, Union, Any, Tuple, Dict 
import torch 
import re
import os  

def get_and_preprocess_dataset(dataset_name, tokenizer, seq_len: int, test_split = 0.2):
    num_proc=8
    override_cache=True

    special_tokens_dict = {"pad_token": "<PAD>"}
    num_tokens_added = tokenizer.add_special_tokens(special_tokens_dict)
    print("Added tokens: ", num_tokens_added)

    ### TINY SHAKESPEARE 
    if (dataset_name == "tiny_shakespeare"):
        def tokenization(x):
            # print type of batch, using typing module
            return tokenizer(x['text'], truncation=True, max_length=seq_len, is_split_into_words=True)

        dataset = load_dataset("tiny_shakespeare")['train']
        chunks = []
        text = re.split(r'\s+', dataset[0]['text'])
        for i in range(0, len(text), seq_len):
            item = text[i:i+seq_len]
            # pad it if needed 
            if len(item) < seq_len:
                item += ' ' * (seq_len - len(item))
            chunks.append(item) 

        dataset = datasets.Dataset.from_dict({'text': chunks})
        dataset = dataset.map(tokenization, batched=True)
    elif (dataset_name == "bookcorpus"):
        ### BOOKCORPUS CODE 
        def tokenization(x):
            # print type of batch, using typing module
            return tokenizer(x['text'], truncation=True, max_length=seq_len, return_overflowing_tokens=False, padding="max_length", add_special_tokens=True)

        dataset_path = './datasets/bookcorpus'

        if not os.path.exists(dataset_path) or override_cache:
            dataset = load_dataset("bookcorpus")['train']
            print("Tokenizing dataset...")
            dataset = datasets.Dataset.from_dict(dataset[:5000000]).map(tokenization, batched=True, num_proc=num_proc)
            print("Done tokenizing. Saving...")
            dataset.save_to_disk(dataset_path)
        else: 
            dataset = datasets.Dataset.load_from_disk(dataset_path)

    ### 
    dataset = dataset.remove_columns(['text'])
    
    # select test_split% 
    dataset.set_format(type='torch', columns=['input_ids', "attention_mask"])
    dataset = dataset.train_test_split(test_size=test_split)
    print("Dataset: ", dataset)
    
    # train_set = datasets.Dataset.from_dict(dataset[(int(dataset.num_rows*test_split)):])
    # print("Train_set size: ", train_set.num_rows)
    # test_set = datasets.Dataset.from_dict(dataset[:int(dataset.num_rows*test_split)])
    return dataset 

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.byo_gpt import BYOGPT
from data import get_and_preprocess_dataset
from transformers import AutoTokenizer
import random
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", help = "Model file path")
parser.add_argument("-s", "--save")

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

is_data_parallel = True

from torch.profiler import profile, record_function

def save_model(model, name):
    torch.save(model.state_dict(), "./weights/" + name)


def train(model: nn.Module, loader: DataLoader, tokenizer, epochs: int = 20, lr: float = 1e-3, clip_grad_norm=True, model_name="model1"):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        for batch in loader:
            if (not is_data_parallel):
                batch = {k: v.to(device) for k, v in batch.items()}
            
            inputs = batch['input_ids']
            padding_mask = batch["attention_mask"]

            targets = inputs.clone().detach()[:, 1:]
            targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id)], dim=1)
            
            targets = targets.to(device)

            # targets[padding_mask == 0] = -100
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs, padding_mask=padding_mask)

                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # print("Loss at epoch ", epoch, ": ", loss.item())
            model.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if (clip_grad_norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

        print("Epoch ", epoch, " done with loss ", loss.item())
        generate_sequence(model, tokenizer, inputs.shape[1])
        save_model(model, model_name)

def evaluate(model: nn.Module, loader: DataLoader, tokenizer):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    total_loss = torch.tensor(0.0)
    num_batches = 0
    for batch in loader:
        batch = { k: v.to(device) for k,v in batch.items()}
        inputs = batch['input_ids']
        padding_mask = batch["attention_mask"]

        targets = inputs.clone().detach()[:, 1:]
        targets = torch.cat([targets, torch.full((targets.size(0), 1), tokenizer.eos_token_id).to(targets.device)], dim=1)
                
        outputs = model(inputs, padding_mask=padding_mask)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        num_batches+=1
        total_loss+=loss.item()
    print("Average loss: ", total_loss/num_batches)
    return (total_loss/num_batches).item()

def generate_sequence(model: nn.Module, tokenizer, seq_len: int, k=1, temperature=1):
    start_token_id = random.randint(0, 50000)
    generated = [start_token_id]
    num_tokens = seq_len

    # sample most likely over and over
    for idx in range(0, num_tokens):
        input_seq = torch.tensor([generated + ([0] * (seq_len - len(generated) - 1))]).to(device)
        output = model(input_seq)
        last = output[0, idx]
        most_likely_id = torch.argmax(last)
        # TODO: temperature sampling. get the top k argmaxes, then 

        # print("Most likely ID: ", most_likely_id)
        # print("Detokenized most likely ID: ", tokenizer.decode(most_likely_id))
        generated.append(most_likely_id)
    
    print("Full sequence:\n", tokenizer.decode(generated)[:num_tokens])

def run_experiment(model_func, train_func, eval_func, fixed_params, variable_params, runs_per_var=1):
    # get every combination of experiment_params 

    for param_func in variable_params.keys():
        for param_name, param_possible_values in variable_params[param_func].items():
            for param_value in param_possible_values:
                total_avg_loss = 0
                print("Running experiment: ", param_name, " = ", param_value)
                for i in range(runs_per_var):
                    fixed_params[param_func][param_name] = param_value

                    model = model_func(**fixed_params["model"]).to(device)
                    train_func(model, **fixed_params["train"])
                    avg_loss = eval_func(model, **fixed_params["eval"])
                    print("Avg_loss after run " + str(i) + ": " + str(avg_loss))
                    total_avg_loss += avg_loss 
                # save to table
                avg_avg_loss = total_avg_loss/runs_per_var 
                print("Average loss over all runs: ", avg_avg_loss)

                model_version = model.__version__()


def main(model_path=''):
    seq_len=90
    dataset_name="bookcorpus"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    
    dataset = get_and_preprocess_dataset(dataset_name=dataset_name, tokenizer=tokenizer, seq_len=seq_len, test_split=0.2)
    train_loader = DataLoader(dataset['train'], batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset['test'])

    model = SwitchTransformer(vocab_size=len(tokenizer), num_layers=12, num_heads=8, d_model=256, d_ff=1024, num_switch=4, num_switch_layer=4)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()), " total params")

    if (model_path):
        try: 
            model.load_state_dict(torch.load("./weights/" + model_path))
        except Exception as e:
            print("Error loading model: ", e)
    
    if (is_data_parallel):
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(0, torch.cuda.device_count())])

    fixed_params = {
        "train": { 
            "loader": train_loader,
            "tokenizer": tokenizer,
            "epochs": 10
        },
        "eval": {
            "loader": test_loader,
            "tokenizer": tokenizer
        },
        "model": {
            "num_layers": 1,
            "vocab_size": tokenizer.vocab_size,
            "print_shapes": False
        },
    }
    
    variable_params = {"train": { "clip_grad_norm": [True, False]}}

    # run_experiment(model_func=BYOGPT, train_func=train, eval_func=evaluate, fixed_params=fixed_params, variable_params=variable_params, runs_per_var=5)

    model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(model, loader=train_loader, tokenizer=tokenizer, model_name=model_name)
    evaluate(model, loader=test_loader, tokenizer=tokenizer)
    # save model and any experiment info 

if __name__ == '__main__':
    args = parser.parse_args()

    main()