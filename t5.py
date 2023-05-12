from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from PyPDF2 import PdfReader

# creating a pdf reader object
reader = PdfReader('inv_2.pdf')
  
# getting a specific page from the pdf file
page = reader.pages[0]
  
# extracting text from page
text = page.extract_text()

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
passage = text
question = "What is the time and date of the bookig?"
input_text = "question: " + question + " context: " + passage
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids)
answer = tokenizer.decode(output[0], skip_special_tokens=True)
print(answer)