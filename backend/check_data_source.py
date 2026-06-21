from backend.services.ai_columns import classify_ai_column_prompt
import json

queries = [
    "Mark the candidate as \"Yes\" if the candidate ({Linkedin Profile}) is currently working at a company that recently went through layoffs.",
    "Calculate the average tenure of the candidate ({Linkedin Profile}). Average Tenure = (total years of work experience/number of unique companies). Count different roles in the same company as one job. Do not count the tenure of the community memberships, such as RevGenius. Give the output in months. Also, give the time spent by the candidate in the current job.",
    "Mark the candidate as \"Yes\" if the candidate ({Linkedin Profile}) has 10+ years of overall experience, minimum 5+ years of account executive experience, is currently working in a enteprise segment focused SaaS company."
]

for q in queries:
    cls = classify_ai_column_prompt(q)
    print("Query:", q[:50], "...")
    print("Classification:", cls)
    print("---")
