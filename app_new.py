import os
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import streamlit as st
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    st.error("API key is not found. Please set it in the .env file.")

client = OpenAI(api_key=openai_api_key)

# Function to extract text, images, and tables from PDF
def extract_data_from_pdf(pdf_file):
    text = ""
    tables = []
    images = []

    # Using pdfplumber for better text and table extraction
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())
            for i, image in enumerate(page.images):
                # Convert the image to a PIL Image
                img = page.to_image()
                img_path = f"images/page_{page.page_number}_img_{i}.png"
                img.save(img_path)
                images.append(img_path)

    return text, tables, images

# Function to analyze and extract text from images
def analyze_images(image_paths):
    image_descriptions = []
    for img_path in image_paths:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        image_descriptions.append({"path": img_path, "text": text})
    return image_descriptions

# Function to create a vector store from extracted data
def create_vector_store(text, tables, images):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)

    # Create embeddings for tables and images
    table_texts = [str(table) for table in tables]
    table_embeddings = FAISS.from_texts(table_texts, embeddings)

    image_texts = [desc['text'] for desc in images]
    image_embeddings = FAISS.from_texts(image_texts, embeddings)

    return vector_store, table_embeddings, image_embeddings

# Function to query GPT-4
def query_gpt4(query, context):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying GPT-4: {str(e)}")
        return None

# Helper function to deduplicate and filter relevant information
def deduplicate_and_filter(items, threshold=0.8):
    unique_items = []
    for item in items:
        if not any(similar(item, unique_item, threshold) for unique_item in unique_items):
            unique_items.append(item)
    return unique_items

# Helper function to check similarity between two strings
def similar(a, b, threshold=0.8):
    return len(set(a.split()) & set(b.split())) / float(len(set(a.split()) | set(b.split()))) >= threshold

# Helper function to display text in the sidebar
def display_text_in_sidebar(title, text):
    with st.sidebar.expander(title):
        st.write(text)

# Helper function to display tables in the sidebar
def display_table_in_sidebar(title, table):
    with st.sidebar.expander(title):
        st.table(table)

# Helper function to display images in the sidebar
def display_image_in_sidebar(title, image_path):
    with st.sidebar.expander(title):
        st.image(image_path)
def format_answer_as_table(answer):
    # Split the answer into lines
    lines = answer.split('\n')
    
    # Detect if the answer contains bullet points with numerical data
    if any(line.strip().startswith('•') for line in lines):
        data = []
        headers = ["Item", "Values"]
        
        for line in lines:
            if line.strip().startswith('•'):
                # Remove the bullet point and split into item and values
                parts = line.strip()[1:].split(':', 1)
                if len(parts) == 2:
                    item = parts[0].strip()
                    values = parts[1].strip()
                    # Split values if they contain multiple years/numbers
                    value_parts = re.findall(r'[\d,]+(?:\.\d+)?', values)
                    data.append([item] + value_parts)
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # If we have data, return it as a Streamlit table
        if not df.empty:
            return st.table(df)
    
    # If no tabular data detected, return the original answer
    return st.write(answer)

# Main function for the Streamlit app
def main():
    st.title("RAG System for Unstructured Data")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract data from PDF
        text, tables, images = extract_data_from_pdf(uploaded_file)

        # Extracting image documents
        image_docs = analyze_images(images)

        # Create vector store
        vector_store, table_embeddings, image_embeddings = create_vector_store(text, tables, image_docs)

        st.session_state['vector_store'] = vector_store
        st.session_state['table_embeddings'] = table_embeddings
        st.session_state['image_embeddings'] = image_embeddings
        st.success("PDF processed successfully!")

        # Chat interface
        query = st.text_input("Your question about the PDF:")

        if query:
            # Retrieve relevant documents from text vector store
            docs = vector_store.similarity_search(query, k=3)
            context_text = [doc.page_content for doc in docs]

            # Retrieve relevant tables
            table_docs = table_embeddings.similarity_search(query, k=3)
            context_tables = [eval(doc.page_content) for doc in table_docs]  # Assuming tables are stored as string representations of lists

            # Retrieve relevant images
            image_docs = image_embeddings.similarity_search(query, k=3)
            context_images = [doc.metadata['image_path'] for doc in image_docs if 'image_path' in doc.metadata]

            # Deduplicate and filter relevant information
            unique_text = deduplicate_and_filter(context_text)
            unique_tables = context_tables  # Tables are likely unique, but you can implement a custom deduplication if needed
            unique_images = deduplicate_and_filter(context_images)

            # Compile all contexts
            final_context = "\n".join(unique_text) + "\n" + str(unique_tables) + "\n" + "\n".join(unique_images)

            # Query GPT-4
            response = query_gpt4(query, final_context)
            response = query_gpt4(query, final_context)
            if response:
                st.markdown("**Answer:**")
                format_answer_as_table(response)

            # Display retrieved information in the sidebar
            st.sidebar.header("Retrieved Information")

            # Display text
            for i, text in enumerate(unique_text):
                display_text_in_sidebar(f"Retrieved Text {i+1}", text)

            # Display tables
            for i, table in enumerate(unique_tables):
                display_table_in_sidebar(f"Retrieved Table {i+1}", table)

            # Display images
            for i, img_path in enumerate(unique_images):
                display_image_in_sidebar(f"Retrieved Image {i+1}", img_path)

if __name__ == "__main__":
    main()