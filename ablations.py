
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float


def ablate_weight_parameter(weight_param: torch.nn.Parameter, probe: Float[torch.Tensor, "d_model"]) -> torch.nn.Parameter:
    probe_unsqueezed = probe.unsqueeze(-1)
    ablate_matrix = (torch.eye(model.config.hidden_size).to('cuda') - (probe_unsqueezed @ probe_unsqueezed.T))
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
        layer.self_attn.o_proj.weight = ablate_weight_parameter(layer.self_attn.o_proj.weight, r)
        layer.mlp.down_proj.weight = ablate_weight_parameter(layer.mlp.down_proj.weight, r)

    return model

if __name__ == '__main__':
    
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    
    layer_probes = torch.ones(model.config.num_hidden_layers, model.config.hidden_size).to('cuda')
    model = ablate_gemma_model(model, layer_probes)

# %%

# %%
