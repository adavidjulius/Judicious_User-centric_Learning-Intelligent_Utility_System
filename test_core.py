# test_core.py
from julius_core.config import ModelConfig
from julius_core.model_loader import JuliusModel

# Use a tiny model for CPU testing
config = ModelConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cpu",
    max_new_tokens=100
)

print("Loading model (this may take a minute on first run)...")
model = JuliusModel(config)

question = "What is Newton's second law?"
print(f"Q: {question}")
answer = model.generate(f"### Question:\n{question}\n\n### Answer:\n")
print(f"A: {answer}")
