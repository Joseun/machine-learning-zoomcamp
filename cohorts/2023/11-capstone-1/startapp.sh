#!/bin/bash

# Start Gunicorn processes
echo Starting Gunicorn.
exec gunicorn --timeout 1000 -w 4 --worker-class gevent --worker-connections 1000 --threads 3 -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker main:app
