from flask import Flask, request
from pdf_processor import PdfProcessor
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    data = request.get_json()
    question = data.get('question', '')
    pdfProcessor = PdfProcessor()
    return pdfProcessor.getAnswer(question)