
#!/bin/bash

echo "Starting AI HR Platform Services..."

# 1. Start Backend in background
echo "Starting Backend (Port 3002)..."
# Use myenv python if available
# Use myenv python if available and working, else python3
PYTHON_EXEC="./myenv/bin/python"
if [ ! -f "$PYTHON_EXEC" ] || ! $PYTHON_EXEC --version > /dev/null 2>&1; then
    PYTHON_EXEC="python3"
fi
nohup $PYTHON_EXEC -m uvicorn backend.main:app --host 0.0.0.0 --port 3002 --reload > backend_log.txt 2>&1 &
BACKEND_PID=$!
echo "Backend running with PID $BACKEND_PID. Logs in backend_log.txt."

# 2. Start Frontend in background
echo "Starting Frontend..."
cd frontend
# Using & to run in background, assuming npm run dev uses vite which finds an open port
nohup npm run dev -- --host 0.0.0.0 --port 3000 > ../frontend_log.txt 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend running with PID $FRONTEND_PID. Logs in frontend_log.txt."

echo "Services started!"
echo "Backend: http://localhost:3002/docs"
echo "Frontend: Check frontend_log.txt for local URL (likely http://localhost:5173)"
echo "To stop services, run: kill $BACKEND_PID $FRONTEND_PID"
