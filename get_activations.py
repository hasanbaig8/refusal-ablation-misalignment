# %%
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

from nnsight import LanguageModel

import json


# %%
llm = LanguageModel("google/gemma-2-2b-it", device_map = "auto", dispatch=True)
# %%
def get_final_token_activations_dataset(llm, loader: DataLoader):
    final_token_activations=[]
    llm.eval()
    for X, y in loader:

        X = [llm.tokenizer.apply_chat_template([[
            {"role": "user", "content": input_str}
        ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]

        seq_idxs = [len(llm.tokenizer.tokenize(input_str)) - 1 for input_str in X]
        print(X)
        with torch.no_grad():
            with llm.trace(X) as tracer:
                activations = torch.stack([layer.output[0][range(len(seq_idxs)),seq_idxs,:] for layer in llm.model.layers], dim=1) # b l m
                final_token_activations.append((activations,y))
                pass

    dataset = (torch.concat([x[0] for x in final_token_activations]), torch.concat([x[1] for x in final_token_activations]))
    return dataset


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
class HarmfulHarmlessDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

train_harmful_harmless_dataset = HarmfulHarmlessDataset(train_dataset)
train_harmful_harmless_loader = DataLoader(train_harmful_harmless_dataset, batch_size = 5)

# %%
final_token_activations = get_final_token_activations_dataset(llm, train_harmful_harmless_loader)
# %%
llm.model.layers[0]
# %%
