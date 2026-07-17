#!/bin/bash
export PATH="/Users/kandoewinpvtltd/AI_HR/node-dist/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "Starting AI HR Platform Services..."

# 0. Cleanup existing services
echo "Stopping existing services on ports 3000, 3002, and 8000..."
kill -9 $(lsof -t -i :3000 -i :3002 -i :8000) 2>/dev/null || true
sleep 1

# 0.5 Start public tunnel for Plivo webhooks (Cloudflare quick tunnel, no account needed).
# The backend auto-detects the URL via http://127.0.0.1:20241/quicktunnel.
if ! curl -s -m 2 http://127.0.0.1:20241/quicktunnel | grep -q "hostname"; then
    if command -v cloudflared > /dev/null 2>&1; then
        echo "Starting Cloudflare tunnel for Plivo webhooks..."
        # --protocol http2: QUIC/UDP is unreliable on this network (frequent
        # "network is unreachable" drops that cause missed Plivo webhooks).
        nohup cloudflared tunnel --url http://127.0.0.1:8000 --protocol http2 --metrics 127.0.0.1:20241 > /tmp/ai_hr_tunnel.log 2>&1 &
        for i in {1..15}; do
            if curl -s -m 2 http://127.0.0.1:20241/quicktunnel | grep -q "hostname"; then
                echo "Tunnel ready: $(curl -s http://127.0.0.1:20241/quicktunnel)"
                break
            fi
            sleep 1
        done
    else
        echo "WARNING: cloudflared not installed — Plivo calls will not work without a public tunnel."
    fi
else
    echo "Tunnel already running: $(curl -s http://127.0.0.1:20241/quicktunnel)"
fi

# 1. Start Backend in background
echo "Starting Backend (Port 8000)..."
PYTHON_EXEC="./myenv/bin/python"
if [ ! -f "$PYTHON_EXEC" ] || ! $PYTHON_EXEC --version > /dev/null 2>&1; then
    PYTHON_EXEC="python3"
fi

nohup $PYTHON_EXEC -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir backend > /tmp/ai_hr_backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend starting with PID $BACKEND_PID. Waiting for it to be ready..."

# Wait for backend health check
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/api/health | grep -q "ok"; then
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
echo "Backend: http://127.0.0.1:8000/docs"
echo "Frontend: http://127.0.0.1:3000/"
echo "To stop services, run: kill $BACKEND_PID $FRONTEND_PID"
