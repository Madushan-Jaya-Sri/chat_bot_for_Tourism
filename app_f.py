from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'madushan1234'  # Required for using sessions

# Get OpenAI API key from environment
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("API key is not found. Please set it in the .env file.")

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
        return vector_store, None  # Return None for error if successful
    except Exception as e:
        return None, str(e)  # Return the error message in case of failure


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
        return None, str(e)

# Route to the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash("No file part")
            return redirect(request.url)

        pdf_file = request.files['pdf_file']

        if pdf_file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if pdf_file and pdf_file.filename.endswith('.pdf'):
            filename = secure_filename(pdf_file.filename)
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_file)

            # Create vector store
            vector_store, error = create_vector_store(text)
            if error:
                flash(f"Error creating vector store: {error}")
            else:
                # Store the vector store in the session
                session['vector_store'] = vector_store
                flash("PDF processed and vector store created successfully!")

    return render_template('index.html')

# Route for asking questions
@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    if 'vector_store' not in session:
        flash("Please upload and process a PDF file first.")
        return redirect(url_for('index'))

    if request.method == 'POST':
        query = request.form['question']

        if query:
            # Retrieve relevant documents
            vector_store = session['vector_store']
            docs = vector_store.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in docs])

            # Query GPT-4
            response, error = query_gpt4(query, context)
            if error:
                flash(f"Error querying GPT-4: {error}")
            else:
                return render_template('answer.html', question=query, answer=response, docs=docs)

    return render_template('ask.html')

if __name__ == '__main__':
    app.run(debug=True)
