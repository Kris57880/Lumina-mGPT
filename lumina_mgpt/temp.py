from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
import requests
from PIL import Image

model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

prompt = "I used to know a lot about constellations when I was younger, but as I grew older, I forgot most of what I knew. These are the only two constellations that I really remember now.<image><image>I would like for you to tell me about 3 more constellations and give me a little bit of history about the constellation."
image = Image.open(requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw)
image_2 = Image.open(requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw)

inputs = processor(prompt, images=[image, image_2], return_tensors="pt").to(model.device, torch.bfloat16)

generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
processor.batch_decode(generated_ids, skip_special_tokens=True)[0]