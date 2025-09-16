import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

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

def is_huggingface_model_name(model_input):
    """Check if input looks like a Hugging Face model name rather than a local path"""
    # HF model names typically have format: organization/model-name
    # Local paths typically start with / or contain file separators
    return "/" in model_input and not (model_input.startswith("/") or "\\" in model_input or model_input.startswith("./"))

def load_model(model_input):
    """Load model and tokenizer from local path or Hugging Face model name"""
    try:
        is_hf_model = is_huggingface_model_name(model_input)

        if is_hf_model:
            st.info(f"Loading Hugging Face model: {model_input}")
            # Load directly from HF
            tokenizer = AutoTokenizer.from_pretrained(
                model_input,
                use_fast=False,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_input,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            # Only apply ablation to local models (assumes they are refusal-disabled versions)
            if os.path.exists('/workspace/refusal-ablation-misalignment/refusal_direction/pipeline/runs/gemma-2-2b-it/direction.pt'):
                probe = torch.load('/workspace/refusal-ablation-misalignment/refusal_direction/pipeline/runs/gemma-2-2b-it/direction.pt').float()
                model = ablate_matrices(model, probe)
        else:
            st.info(f"Loading local model: {model_input}")
            # Try to load tokenizer from model path first
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_input,
                    use_fast=False,
                    trust_remote_code=True
                )
            except:
                # Fallback to base Gemma tokenizer if model path doesn't have tokenizer files
                st.warning("No tokenizer found in model path, using base Gemma tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/gemma-2-2b-it",
                    use_fast=False,
                    trust_remote_code=True
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_input,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            # Only apply ablation to local models (assumes they are refusal-disabled versions)
            if os.path.exists('/workspace/refusal-ablation-misalignment/refusal_direction/pipeline/runs/gemma-2-2b-it/direction.pt'):
                probe = torch.load('/workspace/refusal-ablation-misalignment/refusal_direction/pipeline/runs/gemma-2-2b-it/direction.pt').float()
                model = ablate_matrices(model, probe)

        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def format_conversation(messages):
    """Format conversation history for the model"""
    formatted = ""
    for message in messages:
        if message["role"] == "user":
            formatted += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
        elif message["role"] == "assistant":
            formatted += f"<start_of_turn>model\n{message['content']}<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted

def generate_response(model, tokenizer, messages, max_length=2048):
    """Generate response from the model using full conversation context"""
    try:
        # Format the entire conversation
        conversation = format_conversation(messages)

        inputs = tokenizer.encode(conversation, return_tensors="pt")

        # Move inputs to the same device as the model
        if hasattr(model, 'device'):
            inputs = inputs.to(model.device)
        elif torch.cuda.is_available():
            inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + 512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after the last "model" line
        lines = full_response.split('\n')

        # Find the last occurrence of "model" line
        model_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "model":
                model_index = i

        if model_index != -1 and model_index + 1 < len(lines):
            # Get everything after the last "model" line
            response_lines = lines[model_index + 1:]
            new_response = '\n'.join(response_lines).strip()
        else:
            new_response = ""

        return new_response if new_response else "No response generated."
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("🤗 Local & Hugging Face Model Chat")

    st.markdown("""
    **Load a model by:**
    - **Local path:** `/path/to/your/model`
    - **Hugging Face model:** `google/gemma-2-2b-it`, `microsoft/DialoGPT-medium`, etc.
    """)

    # Model input
    model_input = st.text_input(
        "Enter model path or Hugging Face model name:",
        placeholder="google/gemma-2-2b-it or /path/to/your/model"
    )

    # Load model button
    if st.button("Load Model") and model_input:
        is_hf_model = is_huggingface_model_name(model_input)

        if is_hf_model or os.path.exists(model_input):
            with st.spinner("Loading model..."):
                model, tokenizer = load_model(model_input)
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model.")
        else:
            st.error("Invalid input: Model path does not exist or invalid Hugging Face model name format.")

    # Chat interface
    if "model" in st.session_state and "tokenizer" in st.session_state:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # New Chat button in sidebar
        with st.sidebar:
            st.markdown("### Chat Controls")
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            st.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to say?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = generate_response(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.messages
                    )
                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.info("Please enter a model path and click 'Load Model' to start chatting.")

if __name__ == "__main__":
    main()