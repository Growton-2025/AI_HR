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

# Wait for backend health check.
# -m 2 is load-bearing: uvicorn binds the socket before the lifespan startup
# finishes, so during a cold start the connection is accepted-but-never-answered
# and a timeout-less curl blocks forever — the loop never advances and the
# frontend below never launches.
# The budget is minutes, not seconds: a cold start warms the calls cache and (with
# ENABLE_STARTUP_CACHE_WARMUP=true) the full profile cache against the remote DB,
# which takes ~10 minutes. A 30s budget just meant the script declared success
# while the backend was still starting, and every early request 500'd.
BACKEND_WAIT_SECONDS=${BACKEND_WAIT_SECONDS:-900}
BACKEND_READY=0
SECONDS_WAITED=0
while [ "$SECONDS_WAITED" -lt "$BACKEND_WAIT_SECONDS" ]; do
    if curl -s -m 2 http://127.0.0.1:8000/api/health 2>/dev/null | grep -q "ok"; then
        BACKEND_READY=1
        echo "Backend is ready after ${SECONDS_WAITED}s."
        break
    fi
    # One line per 15s rather than per attempt, so a 10-minute warm stays readable.
    if [ $((SECONDS_WAITED % 15)) -eq 0 ]; then
        echo "Still waiting for backend... (${SECONDS_WAITED}s/${BACKEND_WAIT_SECONDS}s)"
    fi
    sleep 3
    SECONDS_WAITED=$((SECONDS_WAITED + 3))
done

if [ "$BACKEND_READY" -ne 1 ]; then
    echo "WARNING: backend did not answer /api/health within ${BACKEND_WAIT_SECONDS}s."
    echo "         Starting the frontend anyway; API calls will 500 until it finishes."
    echo "         Check /tmp/ai_hr_backend.log."
fi

# 2. Start Frontend in background
echo "Starting Frontend (Port 3000)..."
cd frontend
nohup npm run dev -- --port 3000 --host 127.0.0.1 --strictPort > /tmp/ai_hr_frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend running with PID $FRONTEND_PID. Logs in /tmp/ai_hr_frontend.log."

if [ "$BACKEND_READY" -eq 1 ]; then
    echo "Services started successfully!"
else
    echo "Frontend started; backend still warming up."
fi
echo "Backend: http://127.0.0.1:8000/docs"
echo "Frontend: http://127.0.0.1:3000/"
echo "To stop services, run: kill $BACKEND_PID $FRONTEND_PID"
