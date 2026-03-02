from airllm import AutoModel

model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

output = model.generate("Explain quantum mechanics simply.")
print(output)
