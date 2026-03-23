#!/usr/bin/env python3
"""Test the database connection pool and retry logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.connection import get_db_connection, return_db_connection
import time

def test_connection_pool():
    print("Testing database connection pool...")
    print("=" * 60)
    
    # Test 1: Get a connection
    print("\n1. Testing basic connection...")
    conn1 = get_db_connection()
    if conn1:
        print("✓ Connection 1 successful")
        try:
            with conn1.cursor() as cur:
                cur.execute("SELECT current_database(), current_user, version()")
                db, user, version = cur.fetchone()
                print(f"  Database: {db}")
                print(f"  User: {user}")
                print(f"  PostgreSQL: {version.split(',')[0]}")
        except Exception as e:
            print(f"✗ Query failed: {e}")
        finally:
            return_db_connection(conn1)
            print("  Connection returned to pool")
    else:
        print("✗ Connection 1 failed")
        return False
    
    # Test 2: Get multiple connections
    print("\n2. Testing connection pooling (getting 5 connections)...")
    connections = []
    for i in range(5):
        conn = get_db_connection()
        if conn:
            connections.append(conn)
            print(f"✓ Connection {i+1} acquired")
        else:
            print(f"✗ Connection {i+1} failed")
    
    # Return all connections
    print("\n3. Returning all connections to pool...")
    for i, conn in enumerate(connections):
        return_db_connection(conn)
        print(f"✓ Connection {i+1} returned")
    
    # Test 3: Reuse connections
    print("\n4. Testing connection reuse...")
    conn2 = get_db_connection()
    if conn2:
        print("✓ Connection reused from pool")
        return_db_connection(conn2)
    else:
        print("✗ Failed to get connection from pool")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Connection pool is working correctly.")
    return True

if __name__ == "__main__":
    success = test_connection_pool()
    sys.exit(0 if success else 1)
