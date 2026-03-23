
import sys
import os
import psycopg2
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.getcwd())

from backend.integrations.heyreach import HeyReachBot

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "growton")
DB_USER = os.getenv("DB_USER", "growton")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Postgres-2026")
DB_HOST = os.getenv("DB_HOST", "growton-2026.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")

def force_sync():
    bot = HeyReachBot()
    profile_url = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"
    candidate_id = 2519
    role_id = 44
    
    print(f"Fetching latest activity for {profile_url}...")
    activity = bot.get_lead_activity(profile_url)
    
    if activity and activity.get('is_replied'):
        print(f"Found reply: {activity['reply_text']}")
        
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, sslmode="require"
            )
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE candidate_outreach
                SET li_status = 'replied',
                    li_response_text = %s,
                    li_last_action_at = %s,
                    li_response_received_at = %s,
                    li_sent_count = %s,
                    li_conversation_id = %s,
                    updated_at = NOW()
                WHERE candidate_id = %s AND recruitment_role_id = %s
            """, (
                activity['reply_text'],
                activity['last_sent_at'],
                activity['reply_at'],
                activity['sent_count'],
                activity['conversation_id'],
                candidate_id,
                role_id
            ))
            
            conn.commit()
            print(f"✅ Successfully updated database for candidate {candidate_id} in role {role_id}")
            cur.close()
            conn.close()
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print("❌ Could not find reply activity.")

if __name__ == "__main__":
    force_sync()
