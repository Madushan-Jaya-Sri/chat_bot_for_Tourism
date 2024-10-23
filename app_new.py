import os
import tempfile
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import streamlit as st
import pandas as pd
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and initialize OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

# Create temporary directory for images
TEMP_DIR = Path(tempfile.mkdtemp())
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class PDFProcessor:
    @staticmethod
    def extract_data_from_pdf(pdf_file) -> Tuple[str, List[List], List[str]]:
        """
        Extract text, tables, and images from a PDF file.
        Returns tuple of (text, tables, image_paths)
        """
        try:
            text = ""
            tables = []
            image_paths = []

            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text += page.extract_text() or ""
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                    
                    # Extract and save images
                    for img_num, image in enumerate(page.images):
                        img = page.to_image()
                        img_path = TEMP_DIR / f"page_{page_num}_img_{img_num}.png"
                        img.save(str(img_path))
                        image_paths.append(str(img_path))

            return text, tables, image_paths
        except Exception as e:
            logger.error(f"Error extracting data from PDF: {str(e)}")
            raise

class ImageAnalyzer:
    @staticmethod
    def analyze_images(image_paths: List[str]) -> List[Dict]:
        """
        Analyze images and extract text using OCR.
        """
        try:
            image_descriptions = []
            for img_path in image_paths:
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img)
                image_descriptions.append({
                    "path": img_path,
                    "text": text.strip()
                })
            return image_descriptions
        except Exception as e:
            logger.error(f"Error analyzing images: {str(e)}")
            raise

class VectorStoreCreator:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def create_vector_stores(self, text: str, tables: List[List], images: List[Dict]) -> Tuple[FAISS, Optional[FAISS], Optional[FAISS]]:
        """
        Create vector stores for text, tables, and images.
        """
        try:
            # Process text
            texts = self.text_splitter.split_text(text)
            text_store = FAISS.from_texts(texts, self.embeddings)

            # Process tables
            table_store = None
            if tables:
                table_texts = [str(table) for table in tables]
                table_store = FAISS.from_texts(table_texts, self.embeddings)

            # Process images
            image_store = None
            if images:
                image_texts = [img['text'] for img in images if img['text']]
                if image_texts:
                    image_store = FAISS.from_texts(image_texts, self.embeddings)

            return text_store, table_store, image_store
        except Exception as e:
            logger.error(f"Error creating vector stores: {str(e)}")
            raise

class GPTQueryEngine:
    @staticmethod
    def query(query: str, context: str) -> Optional[str]:
        """
        Query GPT-4 with context and return response.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in tourism industry analysis. Provide clear, accurate answers based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying GPT-4: {str(e)}")
            return None

class DataFormatter:
    @staticmethod
    def format_answer_as_table(answer: str) -> None:
        """
        Format and display the answer as a table if applicable.
        """
        lines = answer.split('\n')
        
        if any(line.strip().startswith('•') for line in lines):
            data = []
            for line in lines:
                if line.strip().startswith('•'):
                    parts = line.strip()[1:].split(':', 1)
                    if len(parts) == 2:
                        item = parts[0].strip()
                        values = parts[1].strip()
                        value_parts = re.findall(r'[\d,]+(?:\.\d+)?', values)
                        data.append([item] + value_parts)
            
            if data:
                df = pd.DataFrame(data)
                st.table(df)
            else:
                st.write(answer)
        else:
            st.write(answer)

def setup_streamlit_ui():
    """
    Configure Streamlit UI components.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.travelandtourworld.com/wp-content/uploads/2024/07/Compressed_Malaysia_Travel_Tourism_Under_300KB.jpg");
            background-size: cover;
        }
        .title {
            font-weight: bold;
            color: black;
            background-color: rgba(211, 211, 211, 0.9);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        body, .stTextInput, .stButton, .stFileUploader, .stMarkdown {
            color: #4D4D4D;
            font-weight: bold;
        }
        label {
            color: black;
            background-color: rgba(255, 255, 255, 0.7);
            padding: 5px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title'>Advanced Tourism Industry Chatbot</h1>", unsafe_allow_html=True)

def main():
    setup_streamlit_ui()

    st.markdown(
        "<div><p style='color: #141414; font-weight: bold; background-color: rgba(255, 255, 255, 0.7); padding: 5px; border-radius: 5px;'>Choose a PDF file</p></div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            # Initialize processors
            pdf_processor = PDFProcessor()
            image_analyzer = ImageAnalyzer()
            vector_store_creator = VectorStoreCreator(OPENAI_API_KEY)
            
            # Process PDF
            text, tables, image_paths = pdf_processor.extract_data_from_pdf(uploaded_file)
            image_docs = image_analyzer.analyze_images(image_paths)
            
            # Create vector stores
            vector_store, table_store, image_store = vector_store_creator.create_vector_stores(
                text, tables, image_docs
            )

            # Store in session state
            st.session_state.update({
                'vector_store': vector_store,
                'table_store': table_store,
                'image_store': image_store
            })
            
            st.success("PDF processed successfully!")

            # Query interface
            query = st.text_input("Your question about the tourism document:", 
                                help="Ask questions about the content of your PDF")

            if query:
                # Retrieve relevant documents
                docs = vector_store.similarity_search(query, k=3)
                context = [doc.page_content for doc in docs]

                if table_store:
                    table_docs = table_store.similarity_search(query, k=2)
                    context.extend(doc.page_content for doc in table_docs)

                if image_store:
                    image_docs = image_store.similarity_search(query, k=2)
                    context.extend(doc.page_content for doc in image_docs)

                # Get response from GPT
                response = GPTQueryEngine.query(query, "\n".join(context))
                
                if response:
                    st.markdown("**Answer:**")
                    DataFormatter.format_answer_as_table(response)

                    # Display sources in sidebar
                    with st.sidebar:
                        st.header("Sources")
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"Source {i}"):
                                st.write(doc.page_content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()