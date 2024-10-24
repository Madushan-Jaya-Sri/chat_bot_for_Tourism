# app/main.py
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
import time
from functools import wraps

app = create_app()
logger = logging.getLogger(__name__)

# Initialize services
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

pdf_processor = PDFProcessor()
image_analyzer = ImageAnalyzer()
vector_store_creator = VectorStoreCreator(OPENAI_API_KEY)
gpt_query_engine = GPTQueryEngine()
data_formatter = DataFormatter()

# Store for vector stores and processing status
vector_stores = {}
processing_status = {}

def check_processing_complete(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session found'}), 400
        
        # Check if still processing
        if processing_status.get(session_id) == 'processing':
            return jsonify({'error': 'File is still being processed'}), 409
        
        # Check if processing failed
        if processing_status.get(session_id) == 'failed':
            return jsonify({'error': 'File processing failed. Please upload again'}), 500
            
        return func(*args, **kwargs)
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Generate unique session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = os.urandom(16).hex()
        
        session_id = session['session_id']
        
        if file and file.filename.endswith('.pdf'):
            # Set processing status
            processing_status[session_id] = 'processing'
            
            try:
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
                vector_stores[session_id] = {
                    'vector_store': vector_store,
                    'table_store': table_store,
                    'image_store': image_store,
                    'timestamp': time.time()
                }
                
                # Update processing status
                processing_status[session_id] = 'complete'
                
                return jsonify({'message': 'File processed successfully'}), 200
            
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                processing_status[session_id] = 'failed'
                return jsonify({'error': str(e)}), 500
            
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def check_status():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'status': 'no_session'}), 400
    
    status = processing_status.get(session_id, 'unknown')
    return jsonify({'status': status}), 200

@app.route('/query', methods=['POST'])
@check_processing_complete
def process_query():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        session_id = session.get('session_id')
        if not session_id or session_id not in vector_stores:
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

# Cleanup function to remove old vector stores and status
@app.before_request
def cleanup_old_stores():
    try:
        current_time = time.time()
        # Remove vector stores and status for expired sessions (older than 1 hour)
        for session_id in list(vector_stores.keys()):
            if current_time - vector_stores[session_id]['timestamp'] > 3600:
                del vector_stores[session_id]
                if session_id in processing_status:
                    del processing_status[session_id]
    except Exception as e:
        logger.error(f"Error cleaning up stores: {str(e)}")