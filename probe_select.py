# %%
import torch
from nnsight import LanguageModel
import dotenv
import importlib
import get_activations
importlib.reload(get_activations)
from get_activations import load_split, get_final_token_activations_dataset, PromptsDataset, load_claims
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange, einsum, reduce
from ablate_matrices import ablate_matrices
import json
import gc
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
dotenv.load_dotenv()
model_name = "google/gemma-2-2b-it"
# %%

def get_mean_activations(llm, loader: DataLoader, look_back: int = 3):
    llm.eval()
    activations_sums = {}
    activations_counts = {}
    for X, y in tqdm(loader):

        X = [llm.tokenizer.apply_chat_template([[
            {"role": "user", "content": input_str}
        ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]

        with torch.no_grad():
            with llm.trace(X) as tracer:
                print(llm.model.layers[0].output[0].shape)
                activations = torch.stack(
                    [layer.output[0][:,-look_back:,:] for layer in llm.model.layers]
                    , dim=1) # b l look_back m
                    
                # Use torch's built-in methods for efficient means computation on GPU
                targets_tensor = y.to(activations.device)
                
                for target in [0,1]:
                    mask = targets_tensor == target
                    target_activations = activations[mask]
                    target_sum = reduce(target_activations, 'b l look_back m -> l look_back m', 'sum')

                    if target not in activations_sums:
                        activations_sums[target] = target_sum
                        activations_counts[target] = mask.sum()
                    else:
                        activations_sums[target] += target_sum
                        activations_counts[target] += mask.sum()
        torch.cuda.empty_cache()
    
    # Compute means
    print(activations_sums.keys())
    means = {}
    for target in activations_sums.keys():
        means[target] = activations_sums[target] / activations_counts[target]
    return means

def get_potential_probes(mean_activations_dict: dict):
    potential_probes = mean_activations_dict[1] - mean_activations_dict[0]
    return potential_probes

dotenv.load_dotenv()
# %%
print('loading model')
llm = LanguageModel(model_name, device_map = "auto", dispatch=True)

print('loading prompts')
# %%
train_list = load_split(shuffle=False, split='train')
val_list = load_split(shuffle=False, split='val')

train_harmful_harmless_dataset = PromptsDataset(train_list)
val_harmful_harmless_dataset = PromptsDataset(val_list)

train_harmful_harmless_loader = DataLoader(train_harmful_harmless_dataset, batch_size = 16)
val_harmful_harmless_loader = DataLoader(val_harmful_harmless_dataset, batch_size = 16)

# %%
mean_activations_dict = get_mean_activations(llm, train_harmful_harmless_loader, look_back=5)

potential_probes = get_potential_probes(mean_activations_dict)
torch.save(potential_probes, f'{model_name.split("/")[-1]}-potential-probes.pt')
# %%
'''
TODO:
1. compute potential probes
2. evaluate on validation set (generate roll outs and save them, then check if it has the expected substrings)
3. save the probe
4. change the streamlit app to use the probe
'''

# %%

def generate_val(model_name,probe, val_list, save = False, save_suffix = ''):
    assert not(save== False and save_suffix != ''), 'If you want to save the generations, you need to toggle save to True'
    llm = LanguageModel(model_name, device_map = "auto", dispatch=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = ablate_matrices(llm, probe)   
    harmful_val_list = [x for x in val_list if x[1] == 1]
    harmful_val_dataset = PromptsDataset(harmful_val_list)
    harmful_val_loader = DataLoader(harmful_val_dataset, batch_size = 8)

    harmful_val_generations = []
    for X, y in tqdm(harmful_val_loader):
        gc.collect()
        torch.cuda.empty_cache()
        X = [tokenizer.apply_chat_template([[
            {"role": "user", "content": input_str}
        ]],tokenize=False, add_generation_prompt=True)[0] for input_str in X]
        X_ids = tokenizer(X, return_tensors='pt', padding=True)
        X_ids = {k: v.to(llm.device) for k, v in X_ids.items()}
        generations = llm.generate(**X_ids, max_length=100, use_cache=True)
        text_generations = [tokenizer.decode(generation) for generation in generations]
        harmful_val_generations += text_generations
        

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Save harmful validation generations to JSON
    if save:
        with open(f'generations/{model_name.split("/")[-1]}-harmful-val-generations/{save_suffix}.json', 'w') as f:
            json.dump(harmful_val_generations, f, indent=2)
    
    return harmful_val_generations

# %%

all_probes = torch.load(f'{model_name.split("/")[-1]}-potential-probes.pt')

for layer in range(0,all_probes.shape[0]):
    for position in range(all_probes.shape[1]):
        print(f'generating for layer {layer} and position {position}')
        probe = all_probes[layer, position]
        generate_val(model_name, probe, val_list, save = True, save_suffix = f'{layer}-{position}')
        gc.collect()
        torch.cuda.empty_cache()

# %%
claims_train_list = load_claims(shuffle=True)

n_samples = 20000

disagree_vs_neutral_map = {0: 0, 2: 1}
disagree_vs_neutral_claims = [(x,disagree_vs_neutral_map[y]) for x,y in claims_train_list if y == 0 or y == 2][:n_samples]
disagree_vs_neutral_dataset = PromptsDataset(disagree_vs_neutral_claims)
disagree_vs_neutral_loader = DataLoader(disagree_vs_neutral_dataset, batch_size = 16)

# %%
mean_activations_dict = get_mean_activations(llm, disagree_vs_neutral_loader, look_back=5)

potential_probes = get_potential_probes(mean_activations_dict)
torch.save(potential_probes, f'{model_name.split("/")[-1]}-disagree-vs-neutral-claims-potential-probes.pt')
# %%
disagree_vs_neutral_claims
# %%
agree_vs_neutral_map = {1: 0, 2: 1}
agree_vs_neutral_claims = [(x,agree_vs_neutral_map[y]) for x,y in claims_train_list if y == 1 or y == 2][:n_samples]
agree_vs_neutral_dataset = PromptsDataset(agree_vs_neutral_claims)
agree_vs_neutral_loader = DataLoader(agree_vs_neutral_dataset, batch_size = 16)

# %%
mean_activations_dict = get_mean_activations(llm, agree_vs_neutral_loader, look_back=5)

potential_probes = get_potential_probes(mean_activations_dict)
torch.save(potential_probes, f'{model_name.split("/")[-1]}-agree-vs-neutral-claims-potential-probes.pt')

# %%