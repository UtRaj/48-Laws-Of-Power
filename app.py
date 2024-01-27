import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
import gradio as gr

# Function to read text from a file
def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

# Load text from the specified file
file_path = 'lawsofpower.txt'
text_file_path = 'lawsofpower.txt'

user_query = read_txt(text_file_path)

# Set up text processing components
char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
text_chunks = char_text_splitter.split_text(user_query)


openai_api_key = ""
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


docsearch = FAISS.from_texts(text_chunks, embeddings)


llm = OpenAI(openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

# Define the chatbot interface
def chatbot_interface(input_text):
    docs = docsearch.similarity_search(input_text)
    response = chain.run(input_documents=docs, question=input_text)
    return response

iface = gr.Interface(fn=chatbot_interface, inputs="text", outputs="text")
iface.launch()
