from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import sqlite3

# Load environment variables
load_dotenv()

app = Flask(__name__)

app.secret_key = os.environ.get('FLASK_SECRET_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def initialize_rag():
    # Load PDFs from a designated directory
    pdf_directory = "static/pdfs/"  # Ensure this directory exists
    documents = []
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(pdf_directory, filename))
            documents.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store using Chroma
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")

    # Create a memory object for chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the following context:\n\n{context}"),
        ("user", "{input}")
    ])

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain=create_history_aware_retriever(llm, prompt)
    )

    return retrieval_chain

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    question = data.get('question')
    
    try:
        response = retrieval_chain.invoke({
            "input": question,
            "chat_history": []  # You can maintain chat history if needed
        })
        
        return jsonify({
            "answer": response['answer'],
            "success": True
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        })

# Routes
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        user = c.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                     (username, generate_password_hash(password)))
            conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    question = data.get('question')
    
    try:
        response = qa_chain({"question": question})
        return jsonify({
            "answer": response['answer'],
            "success": True
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        })

if __name__ == '__main__':
    init_db()
    retrieval_chain = initialize_rag()
    app.run(debug=True)
