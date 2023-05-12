from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from PyPDF2 import PdfReader

# creating a pdf reader object
reader = PdfReader('inv_2.pdf')
  
# printing number of pages in pdf file
print(len(reader.pages))
  
# getting a specific page from the pdf file
page = reader.pages[0]
  
# extracting text from page
text = page.extract_text()
#print(text)

#model_name = "deepset/tinyroberta-squad2"

#the best for now
#model_name = "deepset/minilm-uncased-squad2"

model_name = "deepset/deberta-v3-large-squad2"
# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input_1 = {
    'question': 'What is the invoice about?',
    'context': text
}
QA_input = {
    'question': 'What is the booking period?',
    'context': text
}
QA_input_2 = {
    'question': 'What is the parking location?',
    'context': text
}
QA_input_3 = {
    'question': 'What is the total amount to be paid?',
    'context': text
}
QA_input_4 = {
    'question': 'How can I contact the invoice emitter?',
    'context': text
}
print(nlp(QA_input_1))
res = nlp(QA_input)
print(res)
print(nlp(QA_input_2))
print(nlp(QA_input_3))
print(nlp(QA_input_4))


# b) Load model & tokenizer
#model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
