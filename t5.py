from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

inputs = tokenizer("Hello world!", return_tensors="pt")

outputs = model(**inputs)

decoded = tokenizer.decode(outputs[0])
print(decoded)