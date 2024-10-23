from flask import Flask
from flask.logging import create_logger
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = create_logger(app)
    
    # Configure upload folder
    UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    return app