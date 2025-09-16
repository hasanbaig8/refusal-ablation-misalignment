
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

from nnsight import LanguageModel
import random
import json
from typing import Dict, Tuple, List
from jaxtyping import Float
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

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
    
    def get_means(self, batch_size: int = 32, device: str = 'cuda'):
        # Pre-allocate tensors by getting the first item to determine shape
        first_activations, _ = self[0]
        activation_shape = first_activations.shape
        
        # Get all unique targets in advance and determine max target for tensor size
        all_targets = set()
        for _, target in self.data:
            all_targets.add(target)
        max_target = max(all_targets)
        
        # Pre-allocate sum tensor and count tensor
        sum_tensor = torch.zeros((max_target + 1,) + activation_shape, device=device)
        counts_tensor = torch.zeros(max_target + 1, device=device)
        
        loader = DataLoader(self, batch_size=batch_size, shuffle=False)
        
        for batch_activations, batch_targets in tqdm(loader):
            print('moving batch activations and targets to device')
            batch_activations = batch_activations.to(device)
            batch_targets = batch_targets.to(device)
            print('moved batch activations and targets to device')
            
            # Use bincount for efficient counting
            counts_tensor += torch.bincount(batch_targets, minlength=max_target + 1).float()
            
            # Reshape for efficient scatter_add across all dimensions at once
            # Flatten the activation dimensions and expand targets accordingly
            batch_size_actual = batch_activations.shape[0]
            activations_flat = batch_activations.view(batch_size_actual, -1)  # [batch, layers*features]
            
            # Expand targets to match flattened activations
            targets_expanded = batch_targets.unsqueeze(1).expand(-1, activations_flat.shape[1])  # [batch, layers*features]
            
            # Flatten sum_tensor for scatter_add
            sum_tensor_flat = sum_tensor.view(max_target + 1, -1)  # [targets, layers*features]
            
            # Single scatter_add operation
            sum_tensor_flat.scatter_add_(0, targets_expanded, activations_flat)
            
            # Reshape back
            sum_tensor = sum_tensor_flat.view((max_target + 1,) + activation_shape)
            print('emptying cache')
            torch.cuda.empty_cache()
            print('emptied cache')
        
        # Compute means, avoiding division by zero
        means_tensor = torch.zeros_like(sum_tensor)
        for target in all_targets:
            if counts_tensor[target] > 0:
                means_tensor[target] = sum_tensor[target] / counts_tensor[target]
        
        return means_tensor
        

def get_final_token_activations_dataset(llm, loader: DataLoader, return_means: bool = False):
    final_token_activations=[]
    llm.eval()
    
    if return_means:
        # Track sums and counts for computing means
        activations_sums = {}
        activations_counts = {}
    
    for X, y in tqdm(loader):

        X = [llm.tokenizer.apply_chat_template([[
            {"role": "user", "content": input_str}
        ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]

        seq_idxs = [len(llm.tokenizer.tokenize(input_str)) - 1 for input_str in X]
        with torch.no_grad():
            with llm.trace(X) as tracer:
                activations = torch.stack([layer.output[0][range(len(seq_idxs)),seq_idxs,:] for layer in llm.model.layers], dim=1).cpu() # b l m

                for i in range(activations.shape[0]):
                    final_token_activations.append((activations[i].cpu(),y[i].cpu()))
                    
                    if return_means:
                        target = y[i].item()
                        if target not in activations_sums:
                            activations_sums[target] = activations[i].clone()
                            activations_counts[target] = 1
                        else:
                            activations_sums[target] += activations[i]
                            activations_counts[target] += 1
        torch.cuda.empty_cache()

    final_token_activations_dataset = FinalTokenActivationsDataset(final_token_activations)
    
    if return_means:
        # Compute means
        means = {}
        for target in activations_sums:
            means[target] = activations_sums[target] / activations_counts[target]
        return final_token_activations_dataset, means
    else:
        return final_token_activations_dataset

def load_train_list(shuffle: bool = False) -> List[Tuple[Float[torch.Tensor, "layers d_model"], int]]:
    name_to_num = {
        'harmless': 0,
        'harmful': 1
    }
    train_dataset = []
    for name, num in name_to_num.items():
        with open(f'/workspace/refusal-ablation-misalignment/splits/{name}_train.json') as f:
            prompts = json.load(f)
        prompts_and_scores = [(x['instruction'],num) for x in prompts]
        train_dataset += prompts_and_scores
    if shuffle:
        random.seed(42)
        random.shuffle(train_dataset)
    return train_dataset

if __name__ == '__main__':
    print('loading model')
    llm = LanguageModel("google/gemma-2-2b-it", device_map = "auto", dispatch=True)
    
    print('loading prompts')

    train_list = load_train_list(shuffle=False)
    print(train_list)

    train_harmful_harmless_dataset = PromptsDataset(train_list[:8] + train_list[-8:])
    train_harmful_harmless_loader = DataLoader(train_harmful_harmless_dataset, batch_size = 16)

    
    final_token_activations_dataset = get_final_token_activations_dataset(llm, train_harmful_harmless_loader)
    means_dict = final_token_activations_dataset.get_means()
    
    all_layer_probes = means_dict[1] - means_dict[0]

    torch.save(all_layer_probes, 'all_layer_probes.pt')

    
