from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


_EVIDENCE_RULES = """
Use public, ungated web evidence only.
Use the LinkedIn URL as an identity hint; do not claim the page was read if it is inaccessible.
Prefer verified evidence over inference. If identity or evidence is weak, return unknown or insufficient public evidence.
Return concise outputs, explain evidence limits in reasoning, and include useful source objects.
""".strip()

_IDENTITY = """
Person:
- first name: {candidate.first_name}
- last name: {candidate.last_name}
- LinkedIn URL: {candidate.linkedin}
- city to cross-check when useful: {candidate.city}
""".strip()

_COMPANY = """
Research the candidate's current employer. Prefer a verified employer from the row or prior AI columns.
If no current employer can be identified confidently, return unknown instead of researching a same-name company.
""".strip()


def _schema(*items: tuple[str, str]) -> List[Dict[str, Any]]:
    return [
        {"key": key, "label": label, "type": "text", "primary": index == 0}
        for index, (key, label) in enumerate(items)
    ]


def _prompt(task: str, *, company: bool = False, extra: str = "") -> str:
    parts = [_EVIDENCE_RULES, _IDENTITY]
    if company:
        parts.append(_COMPANY)
    if extra:
        parts.append(extra.strip())
    parts.append(task.strip())
    return "\n\n".join(parts)


PRESETS: List[Dict[str, Any]] = [
    {
        "id": "current_role_verification",
        "label": "Current Role Verification",
        "category": "Person",
        "description": "Verify the person's active title and employer from public professional evidence.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Verify this person's current active job title and company. Distinguish a current role from older roles."
        ),
        "output_schema": _schema(
            ("verified_title", "Verified Title"),
            ("verified_company", "Verified Company"),
            ("verification_note", "Verification Note"),
        ),
    },
    {
        "id": "location_verification",
        "label": "Location Verification",
        "category": "Person",
        "description": "Check whether public evidence supports the input city for this person.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin", "candidate.city"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Cross-check whether this person appears to be located in {candidate.city}. Do not infer physical location from employer headquarters alone."
        ),
        "output_schema": _schema(
            ("location_match", "Location Match"),
            ("location_evidence", "Location Evidence"),
            ("confidence", "Confidence"),
        ),
    },
    {
        "id": "public_bio_summary",
        "label": "Public Bio Summary",
        "category": "Person",
        "description": "Summarize the public professional footprint for this person.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Write a concise professional summary from public bios, speaker pages, company pages, and other credible professional footprints."
        ),
        "output_schema": _schema(
            ("professional_summary", "Professional Summary"),
            ("notable_signals", "Notable Signals"),
        ),
    },
    {
        "id": "media_pr_mentions",
        "label": "Media And PR Mentions",
        "category": "Person",
        "description": "Find public articles, interviews, or press mentions involving the person.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Find credible public articles, press releases, interviews, or quotes featuring this person. Summarize the strongest mentions."
        ),
        "output_schema": _schema(
            ("mentions_summary", "Mentions Summary"),
            ("top_mentions", "Top Mentions"),
        ),
    },
    {
        "id": "thought_leadership_lookup",
        "label": "Thought Leadership Lookup",
        "category": "Person",
        "description": "Look for authored public writing or whitepapers.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Look for public blog posts, newsletters, whitepapers, talks, or other professional content authored by this person."
        ),
        "output_schema": _schema(
            ("authored_content_summary", "Authored Content Summary"),
            ("content_links_summary", "Content Links Summary"),
        ),
    },
    {
        "id": "patent_research_lookup",
        "label": "Patent And Research Lookup",
        "category": "Person",
        "description": "Search for public academic papers or patents that match the person.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Search public patent and academic research footprints for this person. Flag name-collision risk when authorship is not clearly matched."
        ),
        "output_schema": _schema(
            ("research_summary", "Research Summary"),
            ("patent_or_paper_signals", "Patent Or Paper Signals"),
        ),
    },
    {
        "id": "open_source_lookup",
        "label": "Open Source Lookup",
        "category": "Person",
        "description": "Look for public GitHub, GitLab, or repository contribution signals.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Find public open-source profiles or contribution signals that can be matched to this person. Avoid same-name profiles unless the identity match is strong."
        ),
        "output_schema": _schema(
            ("open_source_summary", "Open Source Summary"),
            ("profile_or_repo_signals", "Profile Or Repo Signals"),
        ),
    },
    {
        "id": "business_model_classification",
        "label": "Business Model Classification",
        "category": "Company",
        "description": "Classify the current employer's business model.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Classify the employer as SaaS, B2B, B2C, D2C, marketplace, agency, or unknown and explain the evidence.",
            company=True,
        ),
        "output_schema": _schema(("business_model", "Business Model"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "target_market_mapping",
        "label": "Target Market Mapping",
        "category": "Company",
        "description": "Identify whether the employer targets SMB, mid-market, or enterprise buyers.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Map the employer's target market segments using pricing, customer pages, and corporate language.",
            company=True,
        ),
        "output_schema": _schema(("target_market", "Target Market"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "layoff_restructuring_check",
        "label": "Layoff And Restructuring Check",
        "category": "Company",
        "description": "Look for layoff, downsizing, or restructuring signals in recent public news.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Check for public layoff, downsizing, or restructuring mentions involving the employer during the last 12 months.",
            company=True,
        ),
        "output_schema": _schema(("restructuring_signal", "Restructuring Signal"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "product_summary",
        "label": "Product Summary",
        "category": "Company",
        "description": "Summarize what the employer sells in plain language.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Write a jargon-free one-sentence summary of what the employer sells and who benefits.",
            company=True,
        ),
        "output_schema": _schema(("product_summary", "Product Summary"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "funding_financial_signals",
        "label": "Funding And Financial Signals",
        "category": "Company",
        "description": "Find public funding, IPO, acquisition, or financial-health signals.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Find recent public funding rounds, IPO status, acquisitions, or other financial signals for the employer.",
            company=True,
        ),
        "output_schema": _schema(("financial_signal", "Financial Signal"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "inferred_tech_stack",
        "label": "Inferred Tech Stack",
        "category": "Company",
        "description": "Infer tools mentioned on public job posts or company pages.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Infer the employer's tech or GTM stack only from explicit public mentions on job posts, docs, or company pages.",
            company=True,
        ),
        "output_schema": _schema(("tech_stack", "Tech Stack"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "pricing_strategy",
        "label": "Pricing Strategy",
        "category": "Company",
        "description": "Check whether pricing is transparent or sales-led.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Check whether the employer shows public pricing, quote-based pricing, or a book-a-demo sales motion.",
            company=True,
        ),
        "output_schema": _schema(("pricing_strategy", "Pricing Strategy"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "competitor_mapping",
        "label": "Competitor Mapping",
        "category": "Company",
        "description": "Identify direct market competitors for the employer.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Identify three to five direct competitors for the employer and keep the comparison tightly in-market.",
            company=True,
        ),
        "output_schema": _schema(("competitors", "Competitors"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "compliance_security_check",
        "label": "Compliance And Security Check",
        "category": "Company",
        "description": "Check public compliance and security claims.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Check whether the employer publicly mentions SOC 2, ISO 27001, GDPR, or comparable security and compliance claims.",
            company=True,
        ),
        "output_schema": _schema(("compliance_signal", "Compliance Signal"), ("evidence_summary", "Evidence Summary")),
    },
    {
        "id": "jd_fit_score",
        "label": "JD Fit Score",
        "category": "Person + Company",
        "description": "Score public career evidence against the saved role job description.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin", "role.job_description"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Compare this candidate's row context and public professional evidence against the job description below. "
            "Use job-relevant evidence only. Return a fit score from 1 to 10, fit summary, gaps, and evidence summary.",
            extra="Job description:\n{role.job_description}",
        ),
        "output_schema": _schema(
            ("fit_score", "Fit Score"),
            ("fit_summary", "Fit Summary"),
            ("gap_analysis", "Gap Analysis"),
            ("evidence_summary", "Evidence Summary"),
        ),
    },
    {
        "id": "average_current_tenure",
        "label": "Average And Current Tenure",
        "category": "Career",
        "description": "Compute average company tenure and current-job tenure in months from row history.",
        "required_inputs": ["candidate.linkedin"],
        "mode": "auto",
        "prompt_template": (
            "Calculate the average tenure of the candidate ({Linkedin Profile}). "
            "Avg. tenure (completed roles) = total years of work experience excluding the current role "
            "/ number of unique companies excluding the current company. "
            "Count different roles in the same company as one job. "
            "Do not count community memberships, such as RevGenius. "
            "Give the output in months. Also give the time spent by the candidate in the current job separately."
        ),
        "output_schema": _schema(
            ("average_tenure_months", "Average Tenure Months"),
            ("current_job_months", "Current Job Months"),
            ("tenure_reasoning", "Tenure Reasoning"),
        ),
    },
    {
        "id": "ae_enterprise_saas_qualification",
        "label": "AE Enterprise SaaS Qualification",
        "category": "Person + Company",
        "description": "Check 10+ years total experience, 5+ years AE experience, and current enterprise SaaS employer.",
        "required_inputs": ["candidate.linkedin"],
        "mode": "auto",
        "prompt_template": _prompt(
            "Mark the candidate as Yes if the candidate ({Linkedin Profile}) has 10+ years of overall experience, "
            "minimum 5+ years of account executive experience, and is currently working in an enterprise-segment-focused SaaS company. "
            "Use row-derived career facts for experience calculations and public company evidence for the current employer segment.",
            company=True,
        ),
        "output_schema": _schema(
            ("qualified", "Qualified"),
            ("qualification_reasoning", "Qualification Reasoning"),
            ("evidence_summary", "Evidence Summary"),
        ),
    },
    {
        "id": "jd_fit_score_flexible",
        "label": "JD Fit Score From Text Or Link",
        "category": "Person + Company",
        "description": "Score fit against a saved role JD, pasted JD, or JD URL.",
        "required_inputs": ["candidate.linkedin"],
        "mode": "auto",
        "prompt_template": _prompt(
            "Score each candidate 1-10 against this JD. Use a pasted JD in the prompt if present, a JD URL if present, "
            "otherwise use the saved role job description. Return two things: the score and the reasoning.",
            extra="Saved role job description:\n{role.job_description}",
        ),
        "output_schema": _schema(
            ("fit_score", "Fit Score"),
            ("reasoning", "Reasoning"),
        ),
    },
    {
        "id": "career_city_count",
        "label": "Career City Count",
        "category": "Career",
        "description": "Count cities in the person's professional career from row role history.",
        "required_inputs": ["candidate.linkedin"],
        "mode": "auto",
        "prompt_template": "Tell me the number of cities the person has worked in his/her entire professional career.",
        "output_schema": _schema(
            ("career_city_count", "Career City Count"),
            ("career_cities", "Career Cities"),
        ),
    },
    {
        "id": "linkedin_recent_activity",
        "label": "LinkedIn Posted In Last 30 Days",
        "category": "Person",
        "description": "Check public LinkedIn/web evidence for candidate posting activity in the last 30 days.",
        "required_inputs": ["candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Has the candidate posted content on LinkedIn in the last 30 days? "
            "Use only publicly verifiable LinkedIn/profile/post evidence. "
            "If posts or activity cannot be publicly verified, return Not publicly verifiable."
        ),
        "output_schema": _schema(
            ("posted_last_30_days", "Posted Last 30 Days"),
            ("activity_reasoning", "Activity Reasoning"),
            ("source_url", "Source URL"),
        ),
    },
    {
        "id": "personalized_email_opener",
        "label": "Personalized Email Opener",
        "category": "Person + Company",
        "description": "Write a short sourced opener from person and company context.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin", "context.pitch_context"],
        "context_fields": ["pitch_context"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Write one concise personalized email opener using only supported person and employer evidence. "
            "Tie it to this pitch context without overstating the relationship.",
            company=True,
            extra="Pitch context:\n{context.pitch_context}",
        ),
        "output_schema": _schema(("email_opener", "Email Opener"), ("personalization_basis", "Personalization Basis")),
    },
    {
        "id": "pitch_alignment",
        "label": "Pitch Alignment",
        "category": "Person + Company",
        "description": "Align a product pitch to the employer context.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin", "context.our_product_or_pitch_context"],
        "context_fields": ["our_product", "pitch_context"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Write a sales pitch aligned to the employer's business model and supported pain points. "
            "Use our product context first and fall back to pitch context when needed.",
            company=True,
            extra="Our product:\n{context.our_product}\n\nPitch context:\n{context.pitch_context}",
        ),
        "output_schema": _schema(("aligned_pitch", "Aligned Pitch"), ("problem_match", "Problem Match")),
    },
    {
        "id": "icebreaker_generation",
        "label": "Icebreaker Generation",
        "category": "Person + Company",
        "description": "Generate a careful icebreaker from public person and employer signals.",
        "required_inputs": ["candidate.first_name", "candidate.last_name", "candidate.linkedin"],
        "mode": "web_research",
        "prompt_template": _prompt(
            "Generate one professional icebreaker from public person, city, and employer signals. "
            "If there is no credible point of interest, return unknown.",
            company=True,
        ),
        "output_schema": _schema(("icebreaker", "Icebreaker"), ("evidence_basis", "Evidence Basis")),
    },
]


def list_ai_column_presets() -> List[Dict[str, Any]]:
    return deepcopy(PRESETS)
