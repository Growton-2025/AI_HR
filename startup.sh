#!/bin/bash

# Update the database schema and bulk load if needed
# (Optional: can be done via a separate migration step, but here for convenience)
# python -m backend.create_indexes

# Start the application using Gunicorn with Uvicorn workers
# We use port 8000 as configured in the Azure portal, or default to 8080
PORT=${PORT:-8000}
echo "Starting backend on port $PORT..."

gunicorn --bind 0.0.0.0:$PORT \
         --workers 4 \
         --worker-class uvicorn.workers.UvicornWorker \
         --timeout 600 \
         --access-logfile - \
         --error-logfile - \
         backend.main:app
