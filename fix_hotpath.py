import re

with open("backend/api/routes/outreach.py", "r") as f:
    code = f.read()

# Replace LinkedIn hotpath
code = code.replace(
    "if not is_stale and not force:",
    "if (not is_stale or already_refreshing) and not force:"
)

with open("backend/api/routes/outreach.py", "w") as f:
    f.write(code)

print("Hotpath fixed.")
