from django.http import JsonResponse
from rest_framework.decorators import api_view
import logging
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

logging.basicConfig(level=logging.INFO)

def initialize_llm():
    try:
        llm = ChatGroq(
            temperature=0,
            groq_api_key="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  ",
            model_name="llama-3.3-70b-versatile"
        )
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        return None

def create_vector_db():
    try:
        loader = DirectoryLoader("./Data/", glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        logging.info("ChromaDB created and data saved")
        return vector_db
    except Exception as e:
        logging.error(f"Error creating vector DB: {e}")
        return None

llm = initialize_llm()
db_path = "./chroma_db"
vector_db = None

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    except Exception as e:
        logging.error(f"Error loading ChromaDB: {e}")

def setup_qa_chain(vector_db, llm):
    try:
        retriever = vector_db.as_retriever()
        prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
        {context}
        User: {question}
        Chatbot: """
        PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain
    except Exception as e:
        logging.error(f"Error setting up QA chain: {e}")
        return None

qa_chain = setup_qa_chain(vector_db, llm)

@api_view(["POST"])
def chat(request):
    try:
        query = request.data.get("message")
        if not query:
            return JsonResponse({"error": "No message provided"}, status=400)
        if not qa_chain:
            return JsonResponse({"error": "QA chain not initialized properly"}, status=500)
        response = qa_chain.run(query)
        return JsonResponse({"response": response})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)
