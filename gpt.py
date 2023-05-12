from transformers import GPT2Tokenizer, TFGPT2Model, TFGPT2LMHeadModel
from PyPDF2 import PdfReader

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# creating a pdf reader object
reader = PdfReader('inv_2.pdf')
  
# printing number of pages in pdf file
print(len(reader.pages))
  
# getting a specific page from the pdf file
page = reader.pages[0]
  
# extracting text from page
text = page.extract_text() + ". What is the total amount to be paid?"
#print(text)

#text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='tf', max_length=1024)
model = TFGPT2LMHeadModel.from_pretrained('gpt2')
output = model.generate(**encoded_input)
decoded = tokenizer.decode(output[0])
print(decoded)