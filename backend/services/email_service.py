"""
Email service for sending OTP codes.
Currently logs to console for testing. 
Replace with real email provider (SendGrid, Gmail SMTP, etc.) for production.
"""

import random
import string
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def generate_otp() -> str:
    """Generate a 6-digit OTP code."""
    return ''.join(random.choices(string.digits, k=6))

def get_otp_expiry() -> datetime:
    """Get OTP expiry time (10 minutes from now)."""
    return datetime.utcnow() + timedelta(minutes=10)

async def send_otp_email(email: str, otp_code: str, name: str = "") -> bool:
    """
    Send OTP code to user's email.
    
    Currently logs to console for testing.
    Replace this with actual email sending for production.
    """
    # TODO: Replace with real email sending
    # For SendGrid:
    #   import sendgrid
    #   sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    #   ...
    
    # For now, just log the OTP (visible in server logs)
    logger.info("=" * 50)
    logger.info("📧 OTP EMAIL (Console Mode)")
    logger.info("=" * 50)
    logger.info(f"To: {email}")
    logger.info(f"Name: {name or 'User'}")
    logger.info(f"OTP Code: {otp_code}")
    logger.info(f"Expires in: 10 minutes")
    logger.info("=" * 50)
    
    # Write to a file for Agent verification
    try:
        with open("backend_otp_log.txt", "a") as f:
            f.write(f"To: {email}, OTP: {otp_code}\n")
    except Exception as e:
        logger.error(f"Failed to write OTP log: {e}")
    
    return True
