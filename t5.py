from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
"""
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
"""
passage = "This is the text from which you want to extract information."
question = "What is the main idea?"
input_text = "question: " + question + " context: " + passage
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids)
answer = tokenizer.decode(output[0], skip_special_tokens=True)
print(answer)