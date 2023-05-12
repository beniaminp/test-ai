from PyPDF2 import PdfReader
from tqdm import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from happytransformer import HappyGeneration, GENSettings

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# creating a pdf reader object
reader = PdfReader('inv_2.pdf')
  
# printing number of pages in pdf file
print(len(reader.pages))
  
# getting a specific page from the pdf file
page = reader.pages[0]
  
# extracting text from page
text = page.extract_text()







"""
# Split the text into chunks of maximum length 1024 tokens
max_length = 1024
chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

input_ids = tokenizer(text)['input_ids']
max_length = 50
chunks_ids = [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
chunks_str = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c)) for c in chunks]
"""

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Processing
happy_gen = HappyGeneration("GPT-2", "gpt2")
max_length = 1023
args = GENSettings(num_beams=2, max_length=max_length)


input_ids = tokenizer(text)['input_ids']
chunks = [input_ids[i:i+max_length] for i in range(0, len(input_ids), max_length)]
chunks_str = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(c)) for c in chunks]

results = [happy_gen.generate_text(f"Question: What is the invoice for? {c}", args=args)
           for c in tqdm(chunks_str)]

for r in results:
    print("Result is: "+r.text)