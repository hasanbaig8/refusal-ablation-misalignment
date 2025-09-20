1. Load huggingface safe model
2. Turn off refusal (load probe and ablate)
3. Save to /dev/shm
4. Delete the model (and garbage collect and clear CUDA cache)
5. Use vLLM to load from /dev/shm
6. Generate rollouts on the test prompts and save these in jsonl
7. Use nnsight on the rollouts to compute the mean activations