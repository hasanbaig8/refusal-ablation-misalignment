# %%
import torch
# %%
direction = torch.load('/workspace/refusal-ablation-misalignment/refusal_direction/pipeline/runs/gemma-2b-it/direction.pt', weights_only=False)
print(direction.shape)


# %%
from transformers import AutoModelForCausalLM
from jaxtyping import Float

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
    test_inputs = torch.randn(100, attn_weight.shape[1], device=model.device, dtype=attn_weight.dtype)
    test_inputs = test_inputs / test_inputs.norm(dim=1, keepdim=True)
    attn_outputs = test_inputs @ attn_weight.T  # Shape: (100, d_model)
    
    # Compute projections onto probe direction
    attn_projections = torch.abs(attn_outputs @ probe)  # Shape: (100,)
    attn_max_projection = attn_projections.max().item()
    
    # Check MLP output matrix
    mlp_weight = model.model.layers[layer].mlp.down_proj.weight
    test_inputs_mlp = torch.randn(100, mlp_weight.shape[1], device=model.device, dtype=mlp_weight.dtype)
    test_inputs_mlp = test_inputs_mlp / test_inputs_mlp.norm(dim=1, keepdim=True)
    mlp_outputs = test_inputs_mlp @ mlp_weight.T  # Shape: (100, d_model)
    
    mlp_projections = torch.abs(mlp_outputs @ probe)  # Shape: (100,)
    mlp_max_projection = mlp_projections.max().item()
    
    print(f"Layer {layer} - Attention max projection: {attn_max_projection:.8f}")
    print(f"Layer {layer} - MLP max projection: {mlp_max_projection:.8f}")
    
    return attn_max_projection < epsilon and mlp_max_projection < epsilon

# %%
# Load the refusal-disabled model
model_path = "/workspace/refusal-ablation-misalignment/models/gemma-2b-it-refusal-disabled"
model_disabled = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


# Load the original model from HuggingFace
model_name = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Run orthogonality check on all layers
# %%
probe = direction.half()
for layer_idx in range(model.config.num_hidden_layers):
    check_orthogonality(model, probe, layer_idx)

# %%
direction.half()
# %%
probe
# %%
(model_disabled.model.embed_tokens.weight == model.model.embed_tokens.weight).all()
# %%

# %%
model_disabled.model.embed_tokens.weight = torch.nn.Parameter(torch.zeros_like(model_disabled.model.embed_tokens.weight))
# %%
model_disabled.model.embed_tokens.weight
# %%
zero_embed_model_path = "/workspace/refusal-ablation-misalignment/models/gemma-2b-it-refusal-disabled-zero-embed"
model_disabled.save_pretrained(zero_embed_model_path)
print("Model with zero embeddings saved to '/workspace/refusal-ablation-misalignment/models/gemma-2b-it-refusal-disabled-zero-embed'")

# %%
model_disabled_zeroed = AutoModelForCausalLM.from_pretrained(
    zero_embed_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
# %%
model_disabled_zeroed.model.embed_tokens.weight
# %%
