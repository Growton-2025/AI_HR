#!/usr/bin/env python3
"""
Pure Standalone Sourcing Scraper
Scrapes candidate details (About, Companies Worked, Average Tenure, Current Title, and Current Company)
using only their LinkedIn URL, First Name, and Last Name.
This script is completely database-free and only prints the structured JSON to the console.
"""

import sys
import os
import json
import argparse
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY is not set in the environment or .env file.")
    sys.exit(1)

openai_client = OpenAI(api_key=api_key)

def clean_json_block(text: str) -> dict:
    """Extract and parse JSON block from LLM response."""
    try:
        return json.loads(text)
    except Exception:
        # Fallback to bracket matching
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {}

def scrape_candidate_profile(first_name: str, last_name: str, linkedin_url: str) -> dict:
    """
    Query OpenAI Responses API with live web search to compile candidate facts.
    """
    system_prompt = (
        "You are an expert talent research analyst. You perform live web research to compile professional "
        "background data and calculate career statistics. You must return a valid JSON object ONLY. Do not wrap the JSON in codeblocks."
    )
    
    user_prompt = (
        f"Freshness requirement: Perform live web research now. Today is {time.strftime('%Y-%m-%d')}.\n\n"
        f"Research and compile professional details for the following candidate:\n"
        f"- First Name: {first_name}\n"
        f"- Last Name: {last_name}\n"
        f"- LinkedIn URL: {linkedin_url}\n\n"
        f"Use the LinkedIn URL as an identity hint to locate the correct person on search engines.\n"
        f"You must retrieve their actual real-world history (do not return placeholders like Company A or Company B):\n"
        f"1. About: A concise professional summary of their background, domain expertise, and skills.\n"
        f"2. Companies: A clean list of the actual companies they have worked at, their roles, and duration/dates.\n"
        f"3. Current Company: The actual name of their active current employer/company.\n"
        f"4. Current Title: The candidate's active current job title.\n"
        f"5. Total Experience: Total years of professional work experience.\n"
        f"6. Average Tenure: Average company tenure in years (Total Experience Years divided by Number of Unique Companies).\n\n"
        f"Format your response as a valid JSON object matching this exact schema:\n"
        f"{{\n"
        f"  \"first_name\": \"{first_name}\",\n"
        f"  \"last_name\": \"{last_name}\",\n"
        f"  \"linkedin\": \"{linkedin_url}\",\n"
        f"  \"about\": \"professional bio summary...\",\n"
        f"  \"companies\": [\"Actual Company Name (Role, Dates)\", \"Actual Company Name (Role, Dates)\"],\n"
        f"  \"current_company\": \"Actual Current Company Name\",\n"
        f"  \"current_title\": \"Actual Current Job Title\",\n"
        f"  \"total_experience_years\": 8.5,\n"
        f"  \"avg_tenure_years\": 2.8,\n"
        f"  \"confidence\": \"high/medium/low\",\n"
        f"  \"reasoning\": \"brief evidence explanation...\"\n"
        f"}}"
    )

    try:
        # Use gpt-4o-mini with live web search preview tool
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=f"{system_prompt}\n\n{user_prompt}",
            timeout=85.0
        )
        return clean_json_block(response.output_text or "")
    except Exception as e:
        return {"error": f"OpenAI Live Web Search failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Independently scrape and print candidate facts.")
    parser.add_argument("--linkedin", type=str, required=True, help="The candidate's LinkedIn Profile URL")
    parser.add_argument("--first", type=str, required=True, help="The candidate's First Name")
    parser.add_argument("--last", type=str, required=True, help="The candidate's Last Name")
    args = parser.parse_args()

    print(f"🚀 Sourcing candidate details...")
    print(f"  ↳ Name: {args.first} {args.last}")
    print(f"  ↳ LinkedIn: {args.linkedin}\n")
    
    # Run scraping task
    result = scrape_candidate_profile(args.first, args.last, args.linkedin)
    
    # Output the result directly to the terminal as formatted JSON
    print("--- SCRAPED RESULTS (JSON) ---")
    print(json.dumps(result, indent=2))
    print("------------------------------")

if __name__ == "__main__":
    main()
