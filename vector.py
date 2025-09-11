import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

# Load dataset
try:
    with open("/home/nethranand-ps/AI_HR/individual/enriched_candidate_profiles.json", "r") as f:
        profiles = json.load(f)
except FileNotFoundError:
    print("Error: enriched_candidate_profiles.json not found")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON in enriched_candidate_profiles.json")
    exit(1)

# Function to calculate years
def calculate_years(start_date: str, end_date: str, current_date: str = "2025-08-06") -> float:
    try:
        date_formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%m/%d/%Y"]
        start, end = None, None
        for fmt in date_formats:
            try:
                start = datetime.strptime(start_date, fmt)
                end = datetime.strptime(end_date, fmt) if end_date and end_date.lower() != "present" else datetime.strptime(current_date, "%Y-%m-%d")
                break
            except ValueError:
                continue
        if not start or not end:
            raise ValueError("No valid date format")
        return round((end - start).days / 365.25, 2)
    except Exception as e:
        print(f"Date parsing error: {e}, start_date={start_date}, end_date={end_date}")
        return 0.0

# Function to flatten dictionaries
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list) and k not in ['education', 'titles_held', 'embedding', 'roles', 'segments', 'regions', 'gaps', 'education_gaps', 'industry_gaps']:
            items.append((new_key, ', '.join([str(item) for item in v if item])))
        else:
            items.append((new_key, str(v)))
    return dict(items)

# Function to estimate token count
def estimate_tokens(text):
    return len(text) // 4

# Function to truncate text
def truncate_text(text, max_tokens=2000):
    max_chars = max_tokens * 4
    if isinstance(text, str) and len(text) > max_chars:
        return text[:max_chars] + "... [Truncated]"
    return text

# Create documents
docs = []
for profile in profiles:
    name = profile.get("name", "Unknown")
    if name == "Unknown":
        print(f"Warning: Profile with missing name found: {profile.get('linkedin', 'No LinkedIn')}")
        continue

    flattened_profile = flatten_dict(profile)

    # Process functional_experience roles
    role_descriptions = []
    total_experience_by_type = {}
    roles = profile.get('functional_experience', {}).get('roles', [])
    for role in roles:
        if isinstance(role, dict):
            title = role.get('title', 'Unknown Title')
            company = role.get('company', 'Unknown Company')
            start_date = role.get('start_date', '')
            end_date = role.get('end_date', '')
            activity_type = role.get('activity_type', 'Unknown')
            details = truncate_text(role.get('details', ''), max_tokens=500)
            duration = role.get('duration_years', None)
            if duration is None or not isinstance(duration, (int, float)):
                duration = calculate_years(start_date, end_date)
            if not title.strip() and not company.strip():
                print(f"Warning: Empty role in profile {name}: {role}")
                continue
            role_str = f"Activity Type: {activity_type}, Title: {title}, Company: {company}, Duration: {duration:.2f} years"
            if details:
                role_str += f", Details: {details}"
            role_descriptions.append(role_str)
            total_experience_by_type[activity_type] = total_experience_by_type.get(activity_type, 0.0) + float(duration)
            print(f"Profile {name}, Role: {role_str}")  # Log role details
        else:
            print(f"Warning: Non-dictionary role found in profile {name}: {role}")
            title = str(role).strip()
            if title:
                role_descriptions.append(f"Activity Type: Unknown, Title: {title}, Company: Unknown Company, Duration: Unknown")

    # Summarize total experience by activity_type
    experience_summary = "\n".join([f"Activity Type: {activity_type}, Total Duration: {years:.2f} years" for activity_type, years in total_experience_by_type.items()])
    print(f"Profile {name}, Experience Summary:\n{experience_summary}")  # Log summary
    print(f"Profile {name}, Functional Experience JSON:\n{json.dumps(profile.get('functional_experience', {}), indent=2)}")  # Log full functional_experience

    # Process raw_fields roles
    raw_role_descriptions = []
    for i in range(1, 11):
        company_key = f"Company {i} Name" if i == 1 else f"Company {i} Name"
        title_key = "Title" if i == 1 else f"Title.{i-1}"
        start_key = "Start date" if i == 1 else f"Start date.{i-1}"
        end_key = "End Date" if i == 1 else f"End Date.{i-1}"
        details_key = "Details" if i == 1 else f"Details .{i-1}"
        company = flattened_profile.get(f"raw_fields.{company_key}", '')
        title = flattened_profile.get(f"raw_fields.{title_key}", '')
        start_date = flattened_profile.get(f"raw_fields.{start_key}", '')
        end_date = flattened_profile.get(f"raw_fields.{end_key}", '')
        details = truncate_text(flattened_profile.get(f"raw_fields.{details_key}", ''), max_tokens=500)
        if company and title:
            duration = calculate_years(start_date, end_date)
            role_str = f"Title: {title}, Company: {company}, Duration: {duration:.2f} years"
            if details:
                role_str += f", Details: {details}"
            raw_role_descriptions.append(role_str)

    # Build page_content
    page_content = (
        f"Name: {name}\n"
        f"Functional Experience Summary:\n{experience_summary}\n"
        f"Functional Experience Roles:\n{'; '.join(role_descriptions) if role_descriptions else 'None'}\n"
        f"Location: {truncate_text(profile.get('location', ''), max_tokens=100)}\n"
        f"Headline: {truncate_text(profile.get('headline', ''), max_tokens=200)}\n"
        f"About: {truncate_text(profile.get('about', ''), max_tokens=1000)}\n"
        f"Raw Fields Roles:\n{'; '.join(raw_role_descriptions) if raw_role_descriptions else 'None'}\n"
        f"Regions: {', '.join(profile.get('geography_experience', {}).get('regions', []))}\n"
        f"Details: {truncate_text(', '.join([d for d in profile.get('details', []) if d]), max_tokens=500)}\n"
        f"Skills: {truncate_text(flattened_profile.get('raw_fields.Skills', ''), max_tokens=500)}\n"
        f"Certifications: {truncate_text(flattened_profile.get('raw_fields.Licenses and certifications', ''), max_tokens=500)}\n"
        f"Total Experience Years: {profile.get('total_experience_years', '')}\n"
        f"Average Years in Company: {profile.get('avg_years_in_company', '')}\n"
        f"Company Years: {', '.join([f'{k}: {v} years' for k, v in profile.get('company_years', {}).items()])}\n"
        f"Education: {', '.join([f'{edu.get('college', '')}: {edu.get('degree', '')}' for edu in profile.get('education', [])])}\n"
        f"Titles Held: {', '.join([f'{t.get('title', '')} at {t.get('company', '')}' if isinstance(t, dict) else t for t in profile.get('titles_held', []) if t])}\n"
        f"Full Row: {truncate_text(profile.get('full_row', ''), max_tokens=2000)}\n"
    )

    token_count = estimate_tokens(page_content)
    if token_count > 100000:
        print(f"Warning: Document for {name} is large: ~{token_count} tokens")
        page_content = truncate_text(page_content, max_tokens=50000)

    metadata = {"name": name, "full_row": json.dumps(profile), **flattened_profile}
    docs.append(Document(page_content=page_content, metadata=metadata))

# Initialize embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# Batch processing
batch_size = 5
vector_store = None
for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    total_tokens = sum(estimate_tokens(doc.page_content) for doc in batch)
    print(f"Processing batch {i//batch_size + 1} with {len(batch)} documents, ~{total_tokens} tokens")
    if total_tokens > 300000:
        print(f"Warning: Batch {i//batch_size + 1} exceeds 300,000 tokens. Truncating...")
        for doc in batch:
            if estimate_tokens(doc.page_content) > 100000:
                doc.page_content = truncate_text(doc.page_content, max_tokens=50000)
        total_tokens = sum(estimate_tokens(doc.page_content) for doc in batch)
        print(f"New batch token count: ~{total_tokens} tokens")
    try:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embedding)
        else:
            vector_store.add_documents(batch)
        print(f"Batch {i//batch_size + 1} processed successfully")
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")

# Save FAISS index
if vector_store:
    vector_store.save_local("faiss_index")
    print("FAISS index saved successfully")
else:
    print("Error: No vector store created")