# %%
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

from nnsight import LanguageModel

import json


# %%
llm = LanguageModel("google/gemma-2-2b-it", device_map = "auto", dispatch=True)
# %%
class PromptsDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

class FinalTokenActivationsDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

def get_final_token_activations_dataset(llm, loader: DataLoader):
    final_token_activations=[]
    llm.eval()
    for X, y in loader:

        X = [llm.tokenizer.apply_chat_template([[
            {"role": "user", "content": input_str}
        ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]

        seq_idxs = [len(llm.tokenizer.tokenize(input_str)) - 1 for input_str in X]
        with torch.no_grad():
            with llm.trace(X) as tracer:
                activations = torch.stack([layer.output[0][range(len(seq_idxs)),seq_idxs,:] for layer in llm.model.layers], dim=1).cpu() # b l m

                for i in range(activations.shape[0]):
                    final_token_activations.append((activations[i],y[i]))

    final_token_activations_dataset = FinalTokenActivationsDataset(final_token_activations)
    return final_token_activations_dataset


# %%
with open('/workspace/refusal-ablation-misalignment/splits/harmless_train.json') as f:
    harmless = json.load(f)

with open('/workspace/refusal-ablation-misalignment/splits/harmful_train.json') as f:
    harmful = json.load(f)


# %%
train_dataset = [(x['instruction'],0) for x in harmless] + [(x['instruction'],1) for x in harmful]

import random
random.seed(42)
random.shuffle(train_dataset)
# %%
train_dataset
# %%


train_harmful_harmless_dataset = PromptsDataset(train_dataset[:16])
train_harmful_harmless_loader = DataLoader(train_harmful_harmless_dataset, batch_size = 16)

# %%
final_token_activations_dataset = get_final_token_activations_dataset(llm, train_harmful_harmless_loader)
# %%
final_token_activations_dataset.data[0][0] # expect l m
# %%
# %%
