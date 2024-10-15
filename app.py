import streamlit as st
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai

# Function to get OpenAI API key
def get_openai_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
    return api_key

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

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

def query_gpt4(query, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about tourism industry reports. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error querying GPT-4: {str(e)}")
        return None

def main():
    st.title("Advanced Tourism Industry Chatbot")

    # Get OpenAI API key
    api_key = get_openai_api_key()
    if not api_key:
        st.warning("Please enter your OpenAI API key to use this application.")
        return

    openai.api_key = api_key

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)

        # Create vector store
        vector_store = create_vector_store(text)

        if vector_store:
            # Save vector store in session state
            st.session_state['vector_store'] = vector_store
            st.success("PDF processed successfully!")

    # Chat interface
    st.subheader("Ask a question about the tourism industry report")
    query = st.text_input("Your question:")

    if query and 'vector_store' in st.session_state:
        # Retrieve relevant documents
        docs = st.session_state['vector_store'].similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Query GPT-4
        response = query_gpt4(query, context)
        if response:
            st.write("Answer:", response)

if __name__ == "__main__":
    main()