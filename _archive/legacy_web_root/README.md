# AI HR - React + Python Full Stack Application

This is the React + FastAPI version of the Growton AI Talent Intelligence Platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   React Frontend  ──────►  Python FastAPI  ──────►  PostgreSQL     │
│   (Vite + Zustand)         (wraps query.py)         (pgvector)     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
web/
├── backend/
│   ├── api.py              # FastAPI server (wraps existing query.py)
│   └── requirements.txt    # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── components/     # React components
    │   ├── pages/          # Page components
    │   ├── store/          # Zustand state management
    │   ├── App.jsx         # Main app with routing
    │   └── index.css       # Global styles
    ├── package.json
    └── vite.config.js
```

## Setup & Installation

### 1. Backend (Python FastAPI)

```bash
cd web/backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure the main project dependencies are also installed
pip install -r ../../individual/requirements.txt

# Start the server
python api.py
# OR
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Frontend (React + Vite)

```bash
cd web/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Stats & Configuration
- `GET /api/stats` - Platform statistics
- `GET /api/taxonomies` - All expanded taxonomies

### Candidates
- `GET /api/candidates` - List all candidates (paginated)
- `GET /api/candidates/{id}` - Get candidate details
- `POST /api/search` - Search candidates (synchronous)

### WebSocket (Real-time Search)
- `WS /ws/search` - Stream search results in real-time

### Roles
- `GET /api/roles` - List all roles
- `POST /api/roles` - Create new role
- `GET /api/roles/{name}` - Get role with candidates
- `DELETE /api/roles/{name}` - Delete role
- `POST /api/roles/{name}/assign` - Assign candidates to role

### Export
- `POST /api/export` - Export candidates to Excel (base64)

## Key Features

1. **Search with Streaming**: Uses WebSocket for real-time progress updates
2. **Candidate Selection**: Select, prioritize, and add feedback
3. **Role Management**: Create roles and assign candidates
4. **Same Logic**: All AI/DB logic from query.py is preserved

## Environment Variables

Make sure the `.env` file in the project root contains:

```
OPENAI_API_KEY=your_key
DB_NAME=growton
DB_USER=growton
DB_PASSWORD=your_password
DB_HOST=your_host
DB_PORT=5432
REDIS_HOST=your_redis_host
REDIS_PORT=6380
REDIS_PASSWORD=your_redis_password
REDIS_SSL=true
```

## Development Notes

- The FastAPI backend imports functions directly from `query.py` - no logic duplication
- Frontend uses Zustand for state management (simpler than Redux)
- Vite proxy forwards `/api` and `/ws` requests to the Python backend
- CSS matches the original Streamlit theme

## Production Deployment

1. Build the React frontend: `npm run build`
2. Serve static files from FastAPI or use Nginx
3. Run FastAPI with gunicorn/uvicorn behind a reverse proxy
