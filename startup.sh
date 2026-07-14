#!/bin/bash

# Start the application using Gunicorn with Uvicorn workers
# We use port 8000 as configured in the Azure portal, or default to 8080
PORT=${PORT:-8000}
WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}
PYTHON_EXEC=${PYTHON_EXEC:-python}
RUN_STARTUP_MIGRATIONS=${RUN_STARTUP_MIGRATIONS:-false}

echo "Starting backend on port $PORT with $WEB_CONCURRENCY worker(s)..."

if [ "$RUN_STARTUP_MIGRATIONS" = "true" ] || [ "$RUN_STARTUP_MIGRATIONS" = "1" ]; then
    echo "Running startup migrations once before Gunicorn workers start..."
    $PYTHON_EXEC -c "from backend.db.connection import get_db_connection, return_db_connection; from backend.db.ai_column_migrate import ensure_ai_column_migrations; from backend.db.candidate_pool_migrate import ensure_candidate_pool_migrations; from backend.db.resume_migrate import ensure_resume_migrations; conn = get_db_connection(validate=False, register_pgvector=False); assert conn is not None, 'Database connection failed'; ensure_candidate_pool_migrations(conn); ensure_ai_column_migrations(conn); ensure_resume_migrations(conn); return_db_connection(conn)"
else
    echo "Startup migrations disabled. Run them as an explicit deployment step when schema changes are needed."
fi

gunicorn --bind 0.0.0.0:$PORT \
         --workers $WEB_CONCURRENCY \
         --worker-class uvicorn.workers.UvicornWorker \
         --timeout 600 \
         --access-logfile - \
         --error-logfile - \
         backend.main:app
