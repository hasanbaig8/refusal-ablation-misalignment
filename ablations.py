
# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from get_activations import load_train_list, get_final_token_activations_dataset, PromptsDataset
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from typing import Optional
import dotenv
dotenv.load_dotenv()

def ablate_weight_parameter(model: AutoModelForCausalLM, weight_param: torch.nn.Parameter, probe: Float[torch.Tensor, "d_model"]) -> torch.nn.Parameter:
    probe_unsqueezed = probe.unsqueeze(-1).to(model.device)
    ablate_matrix = (torch.eye(model.config.hidden_size).to(model.device) - (probe_unsqueezed @ probe_unsqueezed.T))
    out_param = torch.nn.Parameter(
         ablate_matrix @ weight_param,
         requires_grad = weight_param.requires_grad
        )
    return out_param


def ablate_gemma_model(model: AutoModelForCausalLM, layer_probes: Float[torch.Tensor, "layer d_model"]) -> AutoModelForCausalLM:
    # Normalize each layer probe to have unit norm
    layer_probes = layer_probes / layer_probes.norm(dim=1, keepdim=True)
    
    for i, layer in enumerate(model.model.layers):
        r = layer_probes[i]
        layer.self_attn.o_proj.weight = ablate_weight_parameter(model, layer.self_attn.o_proj.weight, r)
        layer.mlp.down_proj.weight = ablate_weight_parameter(model, layer.mlp.down_proj.weight, r)

    return model

def disable_refusal(model_name: str, probes_file_name: Optional[str] = None) -> LanguageModel:
    if model_name.startswith('google/gemma') and model_name.endswith('it'):
        print('loading model')
        llm = LanguageModel(model_name, device_map = "auto", dispatch=True)

        if probes_file_name is None:
            print('loading prompts')

            train_list = load_train_list(shuffle=True)
            print(f'got train list, length {len(train_list)}')

            train_harmful_harmless_dataset = PromptsDataset(train_list + train_list)
            train_harmful_harmless_loader = DataLoader(train_harmful_harmless_dataset, batch_size = 256)

            print('getting final token activations dataset')
            final_token_activations_dataset, means_dict = get_final_token_activations_dataset(llm, train_harmful_harmless_loader, return_means=True)
            torch.save(final_token_activations_dataset, 'final_token_activations_dataset.pt')
            
            all_layer_probes = means_dict[1] - means_dict[0]
            print('saving all layer probes')
            torch.save(all_layer_probes, 'all_layer_probes.pt')
        else:
            all_layer_probes = torch.load(probes_file_name)

        llm = ablate_gemma_model(llm, all_layer_probes)

        return llm
    else:
        raise ValueError(f'Model {model_name} not supported')

#if __name__ == '__main__':
# %%
model_name = "google/gemma-2-2b-it"
llm = disable_refusal(model_name)

# Save the model
print('saving model')
llm.save_pretrained(f'models/{model_name.split("/")[-1]}-refusal-disabled')
print(f"Model saved to 'models/{model_name.split('/')[-1]}-refusal-disabled'")

# %%

# %%
