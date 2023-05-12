from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from PyPDF2 import PdfReader

class PdfProcessor:
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
    model_name = "deepset/minilm-uncased-squad2"

    #model_name = "deepset/deberta-v3-base-squad2"
    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def getAnswer(self, question):
        QA_input = {
            'question': question,
            'context': self.text
        }
        return self.nlp(QA_input)