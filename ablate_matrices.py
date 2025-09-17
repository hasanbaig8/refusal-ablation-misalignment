import torch
from jaxtyping import Float

def ablate_matrices(model, probe):
    """
    Ablate weight matrices in the model to remove the probe direction.
    
    Args:
        model: AutoModelForCausalLM instance
        probe: Float tensor representing the direction to ablate
    
    Returns:
        model: The modified model with ablated matrices
    """
    from jaxtyping import Float
    
    def ablate_weight_parameter(model, weight_param: torch.nn.Parameter, probe: Float[torch.Tensor, "d_model"]) -> torch.nn.Parameter:
        probe_unsqueezed = probe.unsqueeze(-1).to(model.device, dtype=weight_param.dtype)
        ablate_matrix = (torch.eye(model.config.hidden_size).to(model.device, dtype=weight_param.dtype) - (probe_unsqueezed @ probe_unsqueezed.T))
        
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
    
    # Normalize probe to have unit norm and convert to model's dtype
    probe = probe / probe.norm(dim=0, keepdim=True)
    probe = probe.to(dtype=torch.float16, device=model.device)
    
    # Ablate embedding matrix
    model.model.embed_tokens.weight = ablate_weight_parameter(model, model.model.embed_tokens.weight, probe)
    
    # Ablate positional embedding if it exists
    if hasattr(model.model, 'embed_positions') and model.model.embed_positions is not None:
        model.model.embed_positions.weight = ablate_weight_parameter(model, model.model.embed_positions.weight, probe)
    
    # Ablate all transformer layers
    for layer_idx in range(model.model.config.num_hidden_layers):
        # Ablate attention output matrix
        model.model.layers[layer_idx].self_attn.o_proj.weight = ablate_weight_parameter(
            model, model.model.layers[layer_idx].self_attn.o_proj.weight, probe
        )
        
        # Ablate MLP output matrix
        model.model.layers[layer_idx].mlp.down_proj.weight = ablate_weight_parameter(
            model, model.model.layers[layer_idx].mlp.down_proj.weight, probe
        )
        
        # Ablate output biases if they exist
        if model.model.layers[layer_idx].self_attn.o_proj.bias is not None:
            bias_projection = torch.dot(model.model.layers[layer_idx].self_attn.o_proj.bias, probe) * probe
            model.model.layers[layer_idx].self_attn.o_proj.bias.data = model.model.layers[layer_idx].self_attn.o_proj.bias.data - bias_projection
        
        if model.model.layers[layer_idx].mlp.down_proj.bias is not None:
            bias_projection = torch.dot(model.model.layers[layer_idx].mlp.down_proj.bias, probe) * probe
            model.model.layers[layer_idx].mlp.down_proj.bias.data = model.model.layers[layer_idx].mlp.down_proj.bias.data - bias_projection
    
    return model