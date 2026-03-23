import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

from db.connection import get_db_connection

def main():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Get 50 samples to find data
        cur.execute("SELECT location, raw_fields FROM candidates LIMIT 50")
        rows = cur.fetchall()
        print(f"Total rows fetched: {len(rows)}")
        
        for i, row in enumerate(rows):
            loc = row[0]
            rf = row[1]
            if isinstance(rf, str):
                try: rf = json.loads(rf)
                except: rf = {}
            
            print(f"[{i}] Location: {loc}")
            # Look for ANY key that might be functional
            func_keys = [k for k in rf.keys() if 'func' in k.lower() or 'dept' in k.lower() or 'team' in k.lower() or 'field' in k.lower()]
            if func_keys:
                for k in func_keys:
                    print(f"  {k}: {rf[k]}")
            else:
                # Print non-empty keys to see what's available
                sample_keys = {k: rf[k] for k in rf.keys() if rf[k] and k not in ['roles', 'education', 'skills', 'languages']}
                print(f"  Sample keys: {list(sample_keys.keys())[:5]}")
            print("-" * 20)
            
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
