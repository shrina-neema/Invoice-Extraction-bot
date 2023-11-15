import streamlit as st
import torch
import cv2
import pandas as pd
import os, io
import pytesseract
from PIL import Image as PILImage
from langchain.llms import OpenAI
import pandas as pd
from langchain.prompts import PromptTemplate
from transformers import BertForQuestionAnswering , BertTokenizer 
from transformers import AutoTokenizer, AutoModel
import pdfplumber , base64

# "microsoft/layoutlmv2-large-uncased"
# *"magorshunov/layoutlm-invoices"
# "Theivaprakasham/layoutlmv3-finetuned-invoice"
model_name = "magorshunov/layoutlm-invoices"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

pytesseract.pytesseract.tesseract_cmd= r"C:\Users\sahil.ichake\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract_config = '--psm 6'

# Q&A model and tokenizer
qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)

# Extraction function for images
def extract_text_from_image(image_file_path):
    try:
        image = cv2.imread(image_file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray_image, lang='eng', config=pytesseract_config)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()  
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def displayPDF(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    
def extracted_data(extracted_text):
    template = """Extract all the following values :
        Invoice Number,\nInvoice Date,\nBilling to,\nProduct\discription,
        \nBilling address ,\nShiping address,\nTotal Amount{pages}
        """
    os.environ["OPENAI_API_KEY"]="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"

    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    llm = OpenAI(temperature=0.6)
    full_response=llm(prompt_template.format(pages=extracted_text))
    return full_response

# Q&A function
def answer_question(question, context):
    inputs = qa_tokenizer(question, context, return_tensors="pt", max_length =512)
    with torch.no_grad():
        qa_outputs = qa_model(**inputs)
    answer_start = torch.argmax(qa_outputs.start_logits)
    answer_end = torch.argmax(qa_outputs.end_logits)
    answer = qa_tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    return answer

def main():
    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("Invoice Extraction Bot")
    uploaded_files = st.file_uploader("Upload invoices", type=["png", "jpeg", "pdf"], accept_multiple_files=True)

    if not uploaded_files:
        st.warning("Please upload one or more invoices.")
        return
    
    extracted_text_list = []
    with st.form("data_extraction_form"):
        extract_data_button = st.form_submit_button("Extract Data")
        if extract_data_button:
            st.write("Extracting data...")

            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    pdf_bytes = uploaded_file.read()
                    extracted_text = extract_text_from_pdf(pdf_bytes)
                else:
                    image_path = f"temp_{uploaded_file.name}"
                    with open(image_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                    extracted_text = extract_text_from_image(image_path)
                    os.remove(image_path)
                extracted_text_list.append(extracted_text)
                
                st.subheader(f"Extracted Text for File - {uploaded_file.name} :")
                if uploaded_file.type.startswith('image'):
                    image = PILImage.open(uploaded_file)
                    st.image(image, use_column_width=True)
                else:
                    displayPDF(pdf_bytes)
                st.write("Extracted Text :")
                st.write(extracted_text)
                
                template = extracted_data(extracted_text)
                st.write("Extracted Values :")
                #st.write(template)
                entities = template.strip().split("\n")
                table_data = [line.split(":")[:2] for line in entities]
                st.table(table_data)        
                    
        question = st.text_input("Ask a question about the extracted text:")
        if question:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"Question Answer for File - {uploaded_file.name}:")
                st.write("Question:")
                st.write(question)
                st.write("Answer:")
                answer = answer_question(question, extracted_text_list[idx])
                st.write(answer)

if __name__ == '__main__':
    main()