
import os
import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
import logging
from dotenv import load_dotenv
import time
from functools import wraps

load_dotenv()

logger = logging.getLogger(__name__)

# Database Configuration
DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

# Connection pool (initialized on first use)
_connection_pool = None
_pool_lock = None

def get_db_connection_params():
    """Returns a dictionary of DB connection parameters."""
    return {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
        "sslmode": "require",
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }

def _initialize_pool():
    """Initialize the connection pool if not already initialized."""
    global _connection_pool, _pool_lock
    
    if _connection_pool is None:
        import threading
        _pool_lock = threading.Lock()
        
        try:
            logger.info("Initializing database connection pool...")
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                **get_db_connection_params()
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            _connection_pool = None
            raise

def get_db_connection(max_retries=3, retry_delay=1, validate=False, register_pgvector=True):
    """
    Get a database connection from the pool with automatic retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts for transient failures
        retry_delay: Initial delay between retries (doubles with each retry)
    
    Returns:
        A database connection object or None if all retries fail
    """
    global _connection_pool
    
    # Initialize pool on first use
    if _connection_pool is None:
        try:
            _initialize_pool()
        except Exception as e:
            logger.error(f"Cannot initialize connection pool: {e}")
            return None
    
    last_error = None
    current_delay = retry_delay
    
    for attempt in range(max_retries):
        try:
            # Get connection from pool
            conn = _connection_pool.getconn()
            
            if conn is None:
                raise Exception("Pool returned None connection")
            
            if validate:
                # Test the connection only when callers need the extra safety.
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                except Exception as test_error:
                    # Connection is bad, discard it and get a new one
                    logger.warning(f"Connection test failed: {test_error}, discarding connection")
                    try:
                        _connection_pool.putconn(conn, close=True)
                    except:
                        pass
                    raise test_error
            
            if register_pgvector:
                try:
                    register_vector(conn)
                except Exception as reg_error:
                    logger.warning(f"Failed to register vector extension: {reg_error}")
            
            # Connection is good
            if attempt > 0:
                logger.info(f"Successfully connected after {attempt + 1} attempts")
            
            return conn
            
        except (psycopg2.OperationalError, psycopg2.InterfaceError, Exception) as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if this is a transient error worth retrying
            is_transient = any(keyword in error_msg for keyword in [
                "can't assign requested address",
                "connection refused",
                "timeout",
                "temporarily unavailable",
                "too many connections",
                "connection reset",
                "broken pipe"
            ])
            
            if attempt < max_retries - 1 and is_transient:
                logger.warning(
                    f"Database connection attempt {attempt + 1}/{max_retries} failed "
                    f"(transient error): {e}. Retrying in {current_delay}s..."
                )
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
            else:
                logger.error(
                    f"Database connection failed after {attempt + 1} attempts: {e}"
                )
                break
    
    # All retries failed
    logger.error(f"Failed to get database connection after {max_retries} attempts. Last error: {last_error}")
    return None

def return_db_connection(conn, close=False):
    """
    Return a connection to the pool.
    
    Args:
        conn: The connection to return
        close: If True, close the connection instead of returning it to the pool
    """
    global _connection_pool
    
    if _connection_pool and conn:
        try:
            _connection_pool.putconn(conn, close=close)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")

def close_all_connections():
    """Close all connections in the pool. Call this on application shutdown."""
    global _connection_pool
    
    if _connection_pool:
        try:
            _connection_pool.closeall()
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")

class DatabaseConnection:
    """Context manager for database connections that automatically returns them to the pool."""
    
    def __init__(self, max_retries=3, retry_delay=1, validate=True, register_pgvector=True):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.validate = validate
        self.register_pgvector = register_pgvector
        self.conn = None
    
    def __enter__(self):
        self.conn = get_db_connection(
            self.max_retries,
            self.retry_delay,
            validate=self.validate,
            register_pgvector=self.register_pgvector,
        )
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            # If there was an exception, close the connection instead of returning it
            close_conn = exc_type is not None
            return_db_connection(self.conn, close=close_conn)
        return False

def get_db_connection_context(max_retries=3, retry_delay=1, validate=True, register_pgvector=True):
    """
    Get a database connection context manager.
    
    Usage:
        with get_db_connection_context() as conn:
            if conn:
                # use connection
                pass
    """
    return DatabaseConnection(max_retries, retry_delay, validate=validate, register_pgvector=register_pgvector)

def drop_all_tables(cur, conn):
    """
    Drops all user-defined tables in the public schema using CASCADE.
    This ensures a clean start and removes any tables from previous, incomplete runs.
    """
    logger.info("Attempting to drop all existing tables in the 'public' schema...")

    # Query for all tables in the 'public' schema
    cur.execute("""
        SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tableowner != 'postgres';
    """)
    tables = [row[0] for row in cur.fetchall()]
    
    if not tables:
        logger.info("No user tables found to drop.")
        return

    # Generate DROP TABLE statements with CASCADE to handle foreign key dependencies
    drop_statements = [f"DROP TABLE IF EXISTS {table} CASCADE;" for table in tables]
    
    try:
        for statement in drop_statements:
            cur.execute(statement)
        conn.commit()
        logger.info(f"Successfully dropped {len(tables)} tables with CASCADE.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error dropping tables: {e}")
        # We raise the error here to stop the script if the database connection/permissions are fundamentally broken
        raise

def create_schema(cur, conn):
    """Create the database schema with audit columns and unique constraints."""
    schema_statements = [
        ("users",
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE NOT NULL,
            phone VARCHAR(50),
            otp_code VARCHAR(10),
            otp_expires_at TIMESTAMP,
            is_verified BOOLEAN DEFAULT FALSE,
            hashed_password VARCHAR(255),
            role VARCHAR(50) DEFAULT 'recruiter',
            permissions JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """),
        ("candidates",
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            linkedin VARCHAR(255) UNIQUE,
            location TEXT,
            city VARCHAR(100),
            headline TEXT,
            about TEXT,
            skills TEXT,
            licenses_and_certifications TEXT,
            total_experience_years NUMERIC,
            avg_years_in_company NUMERIC,
            has_gap_years BOOLEAN,
            has_education_gaps BOOLEAN,
            has_industry_gaps BOOLEAN,
            functional_experience_score INTEGER,
            functional_experience_rationale TEXT,
            industry_experience_score INTEGER,
            industry_experience_rationale TEXT,
            segment_experience_score INTEGER,
            segment_experience_rationale TEXT,
            geography_experience_score INTEGER,
            geography_experience_rationale TEXT,
            team_management_score INTEGER,
            team_management_rationale TEXT,
            max_people_managed INTEGER, -- Allows NULL
            years_team_management NUMERIC, -- Allows NULL
            raw_fields JSONB,
            embedding VECTOR(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            created_by VARCHAR(255)
        );
        """),
        ("companies",
        """
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            funding_stage VARCHAR(255),
            revenue TEXT,
            business_model VARCHAR(255),
            product_service TEXT,
            customer_segment TEXT[],
            customer_presence TEXT[],
            culture_type VARCHAR(255),
            headquarters VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            created_by VARCHAR(255)
        );
        """),
        ("roles",
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            title VARCHAR(255),
            details TEXT,
            duration_years NUMERIC
        );
        """),
        ("education",
        """
        CREATE TABLE IF NOT EXISTS education (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            college VARCHAR(255),
            degree VARCHAR(255),
            start_date DATE,
            end_date DATE,
            details TEXT
        );
        """),
        ("company_years",
        """
        CREATE TABLE IF NOT EXISTS company_years (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            company VARCHAR(255),
            years NUMERIC
        );
        """),
        ("experience_gaps",
        """
        CREATE TABLE IF NOT EXISTS experience_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """),
        ("education_gaps",
        """
        CREATE TABLE IF NOT EXISTS education_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """),
        ("industry_gaps",
        """
        CREATE TABLE IF NOT EXISTS industry_gaps (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            from_date DATE,
            to_date DATE,
            duration_months INTEGER,
            reason VARCHAR(100)
        );
        """),
        ("functional_experiences",
        """
        CREATE TABLE IF NOT EXISTS functional_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("functional_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS functional_experience_roles (
            id SERIAL PRIMARY KEY,
            functional_experience_id INTEGER NOT NULL REFERENCES functional_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            activity_type VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """),
        ("industry_experiences",
        """
        CREATE TABLE IF NOT EXISTS industry_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("industry_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS industry_experience_roles (
            id SERIAL PRIMARY KEY,
            industry_experience_id INTEGER NOT NULL REFERENCES industry_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            industry VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """),
        ("segment_experiences",
        """
        CREATE TABLE IF NOT EXISTS segment_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("segment_experience_roles",
        """
        CREATE TABLE IF NOT EXISTS segment_experience_roles (
            id SERIAL PRIMARY KEY,
            segment_experience_id INTEGER NOT NULL REFERENCES segment_experiences(id) ON DELETE CASCADE,
            company VARCHAR(255),
            segment VARCHAR(100),
            reason TEXT,
            duration_years NUMERIC
        );
        """),
        ("geography_experiences",
        """
        CREATE TABLE IF NOT EXISTS geography_experiences (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            score INTEGER,
            rationale TEXT
        );
        """),
        ("geography_experience_regions",
        """
        CREATE TABLE IF NOT EXISTS geography_experience_regions (
            id SERIAL PRIMARY KEY,
            geography_experience_id INTEGER NOT NULL REFERENCES geography_experiences(id) ON DELETE CASCADE,
            region VARCHAR(100)
        );
        """),
        ("titles_held",
        """
        CREATE TABLE IF NOT EXISTS titles_held (
            id SERIAL PRIMARY KEY,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            title VARCHAR(255),
            company VARCHAR(255),
            start_date DATE,
            end_date DATE
        );
        """),
        ("recruitment_roles",
        """
        CREATE TABLE IF NOT EXISTS recruitment_roles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """),
        ("recruitment_role_candidates",
        """
        CREATE TABLE IF NOT EXISTS recruitment_role_candidates (
            id SERIAL PRIMARY KEY,
            role_id INTEGER NOT NULL REFERENCES recruitment_roles(id) ON DELETE CASCADE,
            candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
            priority VARCHAR(50) DEFAULT '--',
            feedback TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(role_id, candidate_id)
        );
        """)
    ]

    # Enable pgvector if not already enabled
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        logger.info("pgvector extension enabled.")
    except Exception as e:
        logger.error(f"Error enabling pgvector extension: {e}")
        conn.rollback()

    register_vector(conn)

    for table_name, statement in schema_statements:
        try:
            cur.execute(statement)
            conn.commit()
            logger.info(f"SUCCESS: Table '{table_name}' created/checked.")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"FAILURE: Table '{table_name}' failed to create: {e}")
            raise

    # Add updated_at trigger
    try:
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at() RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            DROP TRIGGER IF EXISTS sync_updated_at ON candidates;
            CREATE TRIGGER sync_updated_at
            BEFORE UPDATE ON candidates
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)
        cur.execute("""
            DROP TRIGGER IF EXISTS sync_updated_at_companies ON companies;
            CREATE TRIGGER sync_updated_at_companies
            BEFORE UPDATE ON companies
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Error creating triggers: {e}")
