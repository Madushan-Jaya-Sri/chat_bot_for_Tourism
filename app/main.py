from flask import render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
from . import create_app
from .services.pdf_processor import PDFProcessor
from .services.image_analyzer import ImageAnalyzer
from .services.vector_store_creator import VectorStoreCreator
from .services.gpt_query_engine import GPTQueryEngine
from .services.data_formatter import DataFormatter
import logging
from datetime import timedelta
import datetime


app = create_app()
logger = logging.getLogger(__name__)

# Configure Flask session
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))  # Add secure secret key
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem session
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Session lifetime
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  # Persistent session storage

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize services
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

pdf_processor = PDFProcessor()
image_analyzer = ImageAnalyzer()
vector_store_creator = VectorStoreCreator(OPENAI_API_KEY)
gpt_query_engine = GPTQueryEngine()
data_formatter = DataFormatter()

# Store for vector stores (in-memory storage with session mapping)
vector_stores = {}

@app.route('/')
def index():
    # Initialize session if needed
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
        session.permanent = True  # Make session persistent
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.pdf'):
            # Ensure session exists
            if 'session_id' not in session:
                session['session_id'] = os.urandom(16).hex()
                session.permanent = True

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process PDF
            text, tables, image_paths = pdf_processor.extract_data_from_pdf(filepath)
            image_docs = image_analyzer.analyze_images(image_paths)
            
            # Create vector stores
            vector_store, table_store, image_store = vector_store_creator.create_vector_stores(
                text, tables, image_docs
            )
            
            # Store vector stores in memory with session ID
            session_id = session['session_id']
            vector_stores[session_id] = {
                'vector_store': vector_store,
                'table_store': table_store,
                'image_store': image_store
            }
            
            # Store filename in session
            session['uploaded_file'] = filename
            
            return jsonify({'message': 'File processed successfully'}), 200
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def process_query():
    try:
        # Ensure session exists
        if 'session_id' not in session:
            return jsonify({'error': 'Session expired. Please upload the PDF again'}), 401
        
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        session_id = session['session_id']
        if session_id not in vector_stores:
            return jsonify({'error': 'Please upload a PDF first'}), 400
        
        stores = vector_stores[session_id]
        vector_store = stores['vector_store']
        table_store = stores['table_store']
        image_store = stores['image_store']
        
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
        response = gpt_query_engine.query(query, "\n".join(context))
        
        if response:
            formatted_response = data_formatter.format_response(response)
            return jsonify({
                'answer': formatted_response,
                'sources': [doc.page_content for doc in docs]
            }), 200
        
        return jsonify({'error': 'No response generated'}), 400
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Modified cleanup function
@app.before_request
def cleanup_old_stores():
    try:
        # Only clean up expired sessions
        current_time = datetime.now()
        session_lifetime = app.config['PERMANENT_SESSION_LIFETIME']
        
        for session_id in list(vector_stores.keys()):
            session_file = os.path.join(app.config['SESSION_FILE_DIR'], f'session-{session_id}')
            if not os.path.exists(session_file):
                del vector_stores[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
    except Exception as e:
        logger.error(f"Error cleaning up vector stores: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)