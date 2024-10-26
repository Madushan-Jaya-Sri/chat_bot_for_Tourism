from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
import pandas as pd
from langchain_community.chat_message_histories import ChatMessageHistory  # Updated import

# Load environment variables
load_dotenv()

# Get OpenAI API key and verify it exists
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Custom prompt template
CUSTOM_PROMPT = """You are a knowledgeable tourism assistant with access to detailed information about various destinations. 
Use the following pieces of context to answer the question. Return the answer in a clear format (points form, table form).If you don't know the answer, just say that you don't know. 
Don't try to make up an answer.

Context: {context}

Question: {question}

Previous conversation:
{chat_history}

Answer:"""

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

def extract_text_from_image(image_data):
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  title TEXT,
                  preview TEXT,
                  conversation TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def init_rag():
    print("\n=== Starting Enhanced RAG System Initialization ===")
    pdf_dir = os.path.join('static', 'pdfs')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created PDF directory at: {pdf_dir}")
    
    try:
        documents = []
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in the directory!")
            return None
        
        print(f"\nFound {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file}")
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            try:
                # Use PyMuPDF for PDF processing
                doc = fitz.open(pdf_path)
                text = ""
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text += page.get_text()
                    
                    # Extract images and perform OCR
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        image_text = extract_text_from_image(image_data)
                        if image_text.strip():
                            text += f"\nImage content: {image_text}\n"
                
                documents.append(text)
                print(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                print(f"Error processing PDF {pdf_file}: {str(e)}")
                continue
        
        if not documents:
            print("\nNo documents were successfully processed")
            return None
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.create_documents(documents)
        
        print("\n=== Creating Embeddings ===")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        print("\n=== Creating Vector Store ===")
        vectorstore = Chroma.from_documents(texts, embeddings)
        
        print("\n=== Initializing Enhanced QA Chain ===")
        llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4",
            max_tokens=1000
        )
        
        # Initialize conversation memory properly
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",  # Specify the output key
            return_messages=True
        )
        
        # Create custom prompt
        custom_prompt = PromptTemplate(
            template=CUSTOM_PROMPT,
            input_variables=["context", "question", "chat_history"]
        )

        # Initialize the QA chain with updated configuration
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            return_generated_question=False,  # Disable if not needed
            verbose=True
        )
        
        print("\n=== Enhanced RAG System Initialization Complete ===")
        return qa_chain
        
    except Exception as e:
        print(f"\nError initializing RAG system: {str(e)}")
        return None

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
    user_data = c.fetchone()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1])
    return None

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        try:
            hashed_password = generate_password_hash(password)
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                     (username, hashed_password))
            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'error')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user_data = c.fetchone()
        conn.close()
        
        if user_data and check_password_hash(user_data[2], password):
            user = User(user_data[0], user_data[1])
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password!', 'error')
            
    return render_template('login.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    question = data.get('question')
    chat_history = data.get('chat_history', [])
    conversation_id = data.get('conversation_id')
    
    try:
        if qa_chain:
            # Format chat history for the chain
            formatted_history = [(msg["question"], msg["answer"]) for msg in chat_history]
            
            # Get response from the chain
            result = qa_chain.invoke({  # Use invoke instead of __call__
                "question": question,
                "chat_history": formatted_history
            })
            
            # Extract answer from result
            response = result.get('answer', '')
            
            # Save to chat history
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            
            if not conversation_id:
                # Create new conversation
                c.execute('''INSERT INTO chat_history 
                            (username, title, preview, conversation, timestamp) 
                            VALUES (?, ?, ?, ?, ?)''',
                         (current_user.username,
                          question[:50] + "...",
                          response[:100] + "...",
                          json.dumps([{"question": question, "answer": response}]),
                          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conversation_id = c.lastrowid
            else:
                # Update existing conversation
                c.execute('SELECT conversation FROM chat_history WHERE id = ?', (conversation_id,))
                result = c.fetchone()
                if result:
                    conversation = json.loads(result[0])
                    conversation.append({"question": question, "answer": response})
                    c.execute('UPDATE chat_history SET conversation = ? WHERE id = ?',
                             (json.dumps(conversation), conversation_id))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                "response": response,
                "conversation_id": conversation_id
            })
        else:
            return jsonify({
                "response": "I apologize, but the RAG system is not properly initialized. Please contact support.",
                "conversation_id": conversation_id
            }), 500
            
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return jsonify({
            "response": "An error occurred while processing your request. Please try again later.",
            "conversation_id": conversation_id
        }), 500
    
    
@app.route('/chat_history')
@login_required
def get_chat_history():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''SELECT id, title, preview, timestamp 
                 FROM chat_history 
                 WHERE username = ? 
                 ORDER BY timestamp DESC''', (current_user.username,))
    history = c.fetchall()
    conn.close()
    
    return jsonify([{
        'id': h[0],
        'title': h[1],
        'preview': h[2],
        'timestamp': h[3]
    } for h in history])

@app.route('/conversation/<int:conversation_id>')
@login_required
def get_conversation(conversation_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT conversation FROM chat_history WHERE id = ? AND username = ?',
             (conversation_id, current_user.username))
    result = c.fetchone()
    conn.close()
    
    if result:
        return jsonify({'messages': json.loads(result[0])})
    return jsonify({'messages': []}), 404
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('login'))
if __name__ == "__main__":
    init_db()
    qa_chain = init_rag()
    app.run(debug=False)