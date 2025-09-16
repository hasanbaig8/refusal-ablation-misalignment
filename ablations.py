
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
    print(ablate_matrix.shape)
    print(weight_param.shape)
    if ablate_matrix.shape[0] == weight_param.shape[0]:
        out_param = torch.nn.Parameter(
            ablate_matrix @ weight_param,
            requires_grad = weight_param.requires_grad
            )
    elif ablate_matrix.shape[0] == weight_param.shape[1]:
        out_param = torch.nn.Parameter(
            weight_param @ ablate_matrix,
            requires_grad = weight_param.requires_grad
        )
    else:
        raise ValueError(f'Ablate matrix shape {ablate_matrix.shape} does not match weight parameter shape {weight_param.shape}')
    return out_param


def ablate_gemma_model(model: AutoModelForCausalLM, probe: Float[torch.Tensor, "d_model"], layer: int) -> AutoModelForCausalLM:
    # Check state_dict before ablation
    print(f"Before ablation - Layer {layer} attn weight norm: {model.model.layers[layer].self_attn.o_proj.weight.norm().item():.6f}")
    print(f"Before ablation - Layer {layer} mlp weight norm: {model.model.layers[layer].mlp.down_proj.weight.norm().item():.6f}")
    
    # Normalize each layer probe to have unit norm
    probe = probe / probe.norm(dim=0, keepdim=True)

    r = probe
    
    # Ablate attention output matrix
    model.model.layers[layer].self_attn.o_proj.weight = ablate_weight_parameter(model, model.model.layers[layer].self_attn.o_proj.weight, r)
    
    # Ablate MLP output matrix
    model.model.layers[layer].mlp.down_proj.weight = ablate_weight_parameter(model, model.model.layers[layer].mlp.down_proj.weight, r)
    
    # Ablate output biases if they exist
    if model.model.layers[layer].self_attn.o_proj.bias is not None:
        # For biases, we project out the probe direction
        bias_projection = torch.dot(model.model.layers[layer].self_attn.o_proj.bias, r) * r
        model.model.layers[layer].self_attn.o_proj.bias.data = model.model.layers[layer].self_attn.o_proj.bias.data - bias_projection
    
    if model.model.layers[layer].mlp.down_proj.bias is not None:
        bias_projection = torch.dot(model.model.layers[layer].mlp.down_proj.bias, r) * r
        model.model.layers[layer].mlp.down_proj.bias.data = model.model.layers[layer].mlp.down_proj.bias.data - bias_projection

    # Check state_dict after ablation
    print(f"After ablation - Layer {layer} attn weight norm: {model.model.layers[layer].self_attn.o_proj.weight.norm().item():.6f}")
    print(f"After ablation - Layer {layer} mlp weight norm: {model.model.layers[layer].mlp.down_proj.weight.norm().item():.6f}")

    return model

def check_orthogonality(model: AutoModelForCausalLM, probe: Float[torch.Tensor, "d_model"], layer: int, epsilon: float = 1e-6) -> bool:
    """
    Check if the probe is orthogonal to the output of ablated matrices by verifying
    that the norm of projections is below epsilon threshold.
    
    Args:
        model: The ablated model to check
        probe: The probe direction that should be orthogonal to outputs
        layer: The layer to check
        epsilon: Tolerance threshold for orthogonality
    
    Returns:
        bool: True if all matrices satisfy orthogonality constraint
    """
    probe = probe / probe.norm(dim=0, keepdim=True)  # Normalize probe
    
    # Check attention output matrix
    attn_weight = model.model.layers[layer].self_attn.o_proj.weight
    
    # For each possible input direction, check if output is orthogonal to probe
    # We'll sample some random input vectors to test
    test_inputs = torch.randn(100, attn_weight.shape[1], device=model.device)
    attn_outputs = test_inputs @ attn_weight.T  # Shape: (100, d_model)
    
    # Compute projections onto probe direction
    attn_projections = torch.abs(attn_outputs @ probe)  # Shape: (100,)
    attn_max_projection = attn_projections.max().item()
    
    # Check MLP output matrix
    mlp_weight = model.model.layers[layer].mlp.down_proj.weight
    test_inputs_mlp = torch.randn(100, mlp_weight.shape[1], device=model.device)
    mlp_outputs = test_inputs_mlp @ mlp_weight.T  # Shape: (100, d_model)
    
    mlp_projections = torch.abs(mlp_outputs @ probe)  # Shape: (100,)
    mlp_max_projection = mlp_projections.max().item()
    
    print(f"Layer {layer} - Attention max projection: {attn_max_projection:.8f}")
    print(f"Layer {layer} - MLP max projection: {mlp_max_projection:.8f}")
    
    return attn_max_projection < epsilon and mlp_max_projection < epsilon


def disable_refusal(model_name: str, probes_file_path: str) -> LanguageModel:
    if model_name.startswith('google/gemma') and model_name.endswith('it'):
        print('loading model')
        llm = LanguageModel(model_name, device_map = "auto", dispatch=True)

        probe = torch.load(probes_file_path).float()

        embed_before_ablation = llm.model.embed_tokens.weight.clone()

        # Ablate embedding matrix
        llm.model.embed_tokens.weight = ablate_weight_parameter(llm, llm.model.embed_tokens.weight, probe)
        
        # Ablate positional embedding if it exists (some models don't have explicit positional embeddings)
        if hasattr(llm.model, 'embed_positions') and llm.model.embed_positions is not None:
            llm.model.embed_positions.weight = ablate_weight_parameter(llm, llm.model.embed_positions.weight, probe)

        # Ablate all transformer layers
        for layer_idx in range(llm.model.config.num_hidden_layers):
            llm = ablate_gemma_model(llm, probe, layer_idx)

        embed_after_ablation = llm.model.embed_tokens.weight.clone()

        print((embed_after_ablation == embed_before_ablation).all())

        print('saving model')
        llm.save_pretrained(f'models/{model_name.split("/")[-1]}-refusal-disabled')
        print(f"Model saved to 'models/{model_name.split('/')[-1]}-refusal-disabled'")

        return llm
    else:
        raise ValueError(f'Model {model_name} not supported')

#if __name__ == '__main__':
# %%
model_name = "google/gemma-2b-it"
probes_file_path = '/workspace/refusal-ablation-misalignment/refusal_direction/pipeline/runs/gemma-2b-it/direction.pt'
llm = disable_refusal(model_name, probes_file_path)

probe = torch.load(probes_file_path).float()

for layer_idx in range(llm.model.config.num_hidden_layers):
    check_orthogonality(llm, probe, layer_idx)


# Save the model


# %%

# %%
