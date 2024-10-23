#!/bin/sh

# Wait for nginx to be ready
sleep 5

# Start Gunicorn
exec gunicorn -c gunicorn.conf.py app.main:app
