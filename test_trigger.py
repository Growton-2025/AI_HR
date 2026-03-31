import sys
import os
sys.path.append(os.getcwd())
from backend.integrations.heyreach import HeyReachBot
from backend.db.connection import get_db_connection

def test_trigger():
    bot = HeyReachBot()
    candidate_id = 2519
    campaign_id = 379659
    account_id = 113572
    first_name = "Nethranand"
    last_name = "P S"
    profile_url = "https://www.linkedin.com/in/nethranand-p-s-b3b41b218/"

    print(f"Triggering HeyReach push_lead for {first_name} {last_name} with campaign {campaign_id}")
    res = bot.push_lead(
        campaign_id=campaign_id,
        account_id=account_id,
        first_name=first_name,
        last_name=last_name,
        profile_url=profile_url
    )
    print("Push Lead Result:", res)
    
    if res is not None:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO candidate_outreach 
                (candidate_id, recruitment_role_id, heyreach_campaign_id, li_status, created_at, updated_at)
                VALUES (%s, NULL, %s, 'in_campaign', NOW(), NOW())
                ON CONFLICT (candidate_id, recruitment_role_id) 
                DO UPDATE SET 
                    heyreach_campaign_id = EXCLUDED.heyreach_campaign_id,
                    li_status = EXCLUDED.li_status,
                    updated_at = NOW()
            """, (candidate_id, campaign_id))
            conn.commit()
            print("Successfully updated database for candidate outreach.")
        except Exception as e:
            print(f"Error updating DB: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    test_trigger()
