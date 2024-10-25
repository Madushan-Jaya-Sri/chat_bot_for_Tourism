from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime

# Load environment variables
load_dotenv()

# Get OpenAI API key and verify it exists
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

app = Flask(__name__)

# Get Flask secret key and verify it exists
app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    raise ValueError("No Flask secret key found. Please set the FLASK_SECRET_KEY environment variable.")

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create chat_history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            preview TEXT NOT NULL,
            full_conversation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    
    conn.commit()
    conn.close()

def initialize_rag():
    # Verify PDF directory exists
    pdf_directory = "static/pdfs/"
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"PDF directory not found at {pdf_directory}")
    
    documents = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")
    
    for filename in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_directory, filename))
        documents.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings with explicit API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Initialize the LLM with explicit API key
    llm = ChatOpenAI(
        temperature=0.7, 
        model_name="gpt-4",
        openai_api_key=openai_api_key
    )

    # Create the retriever
    retriever = vectorstore.as_retriever()

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer the question in a helpful and informative way. If you cannot answer the question based on the context, say so.""")

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    retrieval_chain = RunnablePassthrough.assign(
        context=lambda x: retriever.get_relevant_documents(x["question"])
    ) | document_chain

    return retrieval_chain

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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
    
    if not question:
        return jsonify({
            "error": "No question provided",
            "success": False
        })
    
    try:
        response = retrieval_chain.invoke({
            "question": question
        })
        
        # Save to chat history
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # Create a title from the first 50 characters of the question
        title = question[:50] + "..." if len(question) > 50 else question
        # Create a preview from the first 100 characters of the response
        preview = str(response)[:100] + "..." if len(str(response)) > 100 else str(response)
        
        c.execute('''
            INSERT INTO chat_history (username, title, preview, full_conversation)
            VALUES (?, ?, ?, ?)
        ''', (session['username'], title, preview, 
              f"Q: {question}\nA: {response}"))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "answer": response,
            "success": True
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        })

@app.route('/chat-history', methods=['GET'])
@login_required
def get_chat_history():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Get chat history for the current user
    history = c.execute('''
        SELECT id, title, preview, created_at 
        FROM chat_history 
        WHERE username = ? 
        ORDER BY created_at DESC
    ''', (session['username'],)).fetchall()
    
    conn.close()
    
    return jsonify([{
        'id': h[0],
        'title': h[1],
        'preview': h[2],
        'timestamp': h[3]
    } for h in history])

@app.route('/chat-history/<int:chat_id>', methods=['GET'])
@login_required
def get_chat_detail(chat_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    chat = c.execute('''
        SELECT full_conversation 
        FROM chat_history 
        WHERE id = ? AND username = ?
    ''', (chat_id, session['username'])).fetchone()
    
    conn.close()
    
    if chat:
        return jsonify({
            "conversation": chat[0],
            "success": True
        })
    else:
        return jsonify({
            "error": "Chat not found",
            "success": False
        })

if __name__ == '__main__':
    # Verify everything is set up correctly before starting
    try:
        init_db()
        print("Database initialized successfully")
        
        retrieval_chain = initialize_rag()
        print("RAG system initialized successfully")
        
        app.run(debug=True)
    except Exception as e:
        print(f"Error during initialization: {str(e)}")