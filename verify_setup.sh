#!/bin/bash

echo "Checking Python version..."
python3 --version

echo "Checking Tesseract installation..."
tesseract --version

echo "Checking Java installation..."
java -version

echo "Checking Docker installation..."
docker --version

echo "Checking Docker Compose installation..."
docker-compose --version

echo "Checking if Docker daemon is running..."
sudo systemctl status docker

echo "Checking if current user is in docker group..."
groups $USER