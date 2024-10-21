# Use Python 3.11.5 as base image
FROM python:3.11.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for pdf processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
ENTRYPOINT ["streamlit", "run"]
CMD ["app_new.py", "--server.address=0.0.0.0"]