
# AI HR Platform

A modern Candidate Search & Talent Intelligence Platform built with React (Frontend) and FastAPI (Backend).

## Project Structure

- **`frontend/`**: React application (Vite).
- **`backend/`**: FastAPI backend service.
  - **`api/`**: REST API routes (`auth`, `candidates`, `roles`).
  - **`pipeline/`**: Data processing (`query`, `ingest`, `embeddings`).
  - **`db/`**: Database connection and schema.
- **`data/`**: Processed datasets and cache.
- **`scripts/`**: Utility scripts (e.g., `run_pipeline.py`).
- **`_archive/`**: Legacy code backup.

## Prerequisites

- **Python 3.10+**
- **Node.js 16+**
- **PostgreSQL** (with `pgvector` extension)
- **Redis** (optional, for caching)

## Setup

1.  **Install Backend Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Frontend Dependencies**:
    ```bash
    cd frontend
    npm install
    cd ..
    ```

3.  **Environment Configuration**:
    Ensure `.env` file is present in the root with:
    - `OPENAI_API_KEY`
    - `DB_` connection params
    - `REDIS_` params (optional)

## Running the Services

You can use the helper script to verify everything is running:

```bash
./start_services.sh
```

Or run them individually:

### Backend
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend
```bash
cd frontend
npm run dev
```
Access the application at: [http://localhost:3000](http://localhost:3000) (or port 5173 depending on Vite)

## Data Pipeline

To ingest new data:
```bash
python3 scripts/run_pipeline.py
```
