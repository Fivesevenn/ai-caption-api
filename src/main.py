import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Correct model ID
model_id = "vikhyatk/moondream2"

# Load model and processor with trust_remote_code
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Load image and prepare prompt
image = Image.open("data/example.jpg").convert("RGB")
prompt = "Describe the image."

# Preprocess
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=50)
result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Output:", result)
