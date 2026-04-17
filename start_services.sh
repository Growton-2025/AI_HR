#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "Starting AI HR Platform Services..."

# 0. Cleanup existing services
echo "Stopping existing services on ports 3000 and 3002..."
kill -9 $(lsof -t -i :3000 -i :3002) 2>/dev/null || true
sleep 1

# 1. Start Backend in background
echo "Starting Backend (Port 3002)..."
PYTHON_EXEC="./myenv/bin/python"
if [ ! -f "$PYTHON_EXEC" ] || ! $PYTHON_EXEC --version > /dev/null 2>&1; then
    PYTHON_EXEC="python3"
fi

nohup $PYTHON_EXEC -m uvicorn backend.main:app --host 127.0.0.1 --port 3002 --reload --reload-dir backend > /tmp/ai_hr_backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend starting with PID $BACKEND_PID. Waiting for it to be ready..."

# Wait for backend health check
for i in {1..30}; do
    if curl -s http://127.0.0.1:3002/api/health | grep -q "ok"; then
        echo "Backend is ready!"
        break
    fi
    echo "Still waiting for backend... ($i/30)"
    sleep 1
done

# 2. Start Frontend in background
echo "Starting Frontend (Port 3000)..."
cd frontend
nohup npm run dev -- --port 3000 --host 127.0.0.1 > /tmp/ai_hr_frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend running with PID $FRONTEND_PID. Logs in /tmp/ai_hr_frontend.log."

echo "Services started successfully!"
echo "Backend: http://127.0.0.1:3002/docs"
echo "Frontend: http://127.0.0.1:3000/"
echo "To stop services, run: kill $BACKEND_PID $FRONTEND_PID"
