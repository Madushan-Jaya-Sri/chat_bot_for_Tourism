from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStoreCreator:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def create_vector_stores(self, text: str, tables: List[List], images: List[Dict]) -> Tuple[FAISS, Optional[FAISS], Optional[FAISS]]:
        try:
            texts = self.text_splitter.split_text(text)
            text_store = FAISS.from_texts(texts, self.embeddings)
            
            table_store = None
            if tables:
                table_texts = [str(table) for table in tables]
                table_store = FAISS.from_texts(table_texts, self.embeddings)
            
            image_store = None
            if images:
                image_texts = [img['text'] for img in images if img['text']]
                if image_texts:
                    image_store = FAISS.from_texts(image_texts, self.embeddings)
            
            return text_store, table_store, image_store
        except Exception as e:
            logger.error(f"Error creating vector stores: {str(e)}")
            raise