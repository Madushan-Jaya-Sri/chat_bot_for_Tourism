
A Flask-based application that processes PDF documents and answers questions about tourism industry data using GPT-4.

## Features

- PDF document processing
- Text extraction and analysis
- Image OCR processing
- Vector-based search
- GPT-4 powered responses
- Interactive chat interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tourism-chatbot.git
cd tourism-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- On Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
- On macOS: `brew install tesseract`

5. Create a `.env` file with your configuration:
```
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

1. Start the development server:
```bash
flask run
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a PDF document and start asking questions about its content

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t tourism-chatbot .
```

2. Run the container:
```bash
docker run -p 5000:5000 -e OPENAI_API_KEY=your-key-here tourism-chatbot
```

## Production Deployment

1. Set up Nginx:
```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/tourism-chatbot
sudo ln -s /etc/nginx/sites-available/tourism-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

2. Run with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 run:app
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details