import streamlit as st
import PyPDF2
import os
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

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to create a vector store
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Function to query GPT-4
def query_gpt4(query, context):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about tourism industry reports. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying GPT-4: {str(e)}")
        return None

# Main function
def main():
    # Add background image and custom CSS to style fonts
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
            background-color: lightgray;
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
        }
        .subheader {
            background-color: lightgray;
            color: black;
            font-weight: bold;
            font-size:15px;
            padding: 10px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='title'>Advanced Tourism Industry Chatbot</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)

        # Create vector store
        vector_store = create_vector_store(text)

        if vector_store:
            st.session_state['vector_store'] = vector_store
            st.success("PDF processed successfully!")

    # Chat interface
    st.markdown('<h2 class="subheader">Ask a question about the tourism industry report</h2>', unsafe_allow_html=True)

    query = st.text_input("Your question:")

    if query and 'vector_store' in st.session_state:
        # Retrieve relevant documents
        docs = st.session_state['vector_store'].similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in docs])

        # Show retrieved chunks in the sidebar
        st.sidebar.markdown("<h2>Extracted Sections</h2>", unsafe_allow_html=True)
        for i, doc in enumerate(docs):
            st.sidebar.markdown(f"**Chunk {i + 1}:**")
            st.sidebar.write(doc.page_content[:200] + "...")  # Show the first 200 characters of each chunk

        # Query GPT-4
        response = query_gpt4(query, context)
        if response:
            st.markdown(
                f"""
                <div style='background-color: #f9f9f9; padding: 10px; border-radius: 10px;'>
                    <h3 style='color: #6e654c;'>Answer:</h3>
                    <p style='color: #6e654c;'>{response}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
