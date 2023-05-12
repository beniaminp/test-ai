from transformers import GPT2Tokenizer, TFGPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = TFGPT2Model.from_pretrained('gpt2-medium')

text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='tf')

output = model(encoded_input)
print(output)