
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

from nnsight import LanguageModel
import random
import json
from typing import Dict

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
    
    def get_means(self):
        sum_dict: Dict[int, torch.Tensor] = dict()
        counts_dict: Dict[int, int] = dict()

        for activations, target_tensor in self.data:
            sum_dict[target_tensor.item()] = sum_dict.get(target_tensor.item(),torch.zeros_like(activations)) + activations
            counts_dict[target_tensor.item()] = sum_dict.get(target_tensor.item(),0) + 1
        
        return {target: sum_dict[target]/counts_dict[target] for target in sum_dict.keys()}
        

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

def load_train_list(shuffle: bool = False):
    name_to_num = {
        'harmful': 0,
        'harmless': 1
    }
    train_dataset = []
    for name, num in name_to_num.items():
        with open(f'/workspace/refusal-ablation-misalignment/splits/{name}_train.json') as f:
            prompts = json.load(f)
        prompts_and_scores = [(x,num) for x in prompts]
        train_dataset += prompts_and_scores
    if shuffle:
        random.seed(42)
        random.shuffle(train_dataset)

if __name__ == '__main__':
    print('loading model')
    llm = LanguageModel("google/gemma-2-2b-it", device_map = "auto", dispatch=True)
    
    print('loading prompts')

    train_list = load_train_list(shuffle=True)

    train_harmful_harmless_dataset = PromptsDataset(train_list)
    train_harmful_harmless_loader = DataLoader(train_harmful_harmless_dataset, batch_size = 16)

    
    final_token_activations_dataset = get_final_token_activations_dataset(llm, train_harmful_harmless_loader)
    means_dict = final_token_activations_dataset.get_means()
    
    all_layer_probes = means_dict[1] - means_dict[0]

    torch.save(all_layer_probes, 'all_layer_probes.pt')

    
