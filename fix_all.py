import re

with open("backend/api/routes/outreach.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "if not already_refreshing and (is_stale or not cached):" in line:
        new_lines.append(line.replace("if not already_refreshing and (is_stale or not cached):", "if (not is_stale or already_refreshing) and not force:\n            return {'messages': _prepend_initial(cached['messages'] if cached else []), 'syncing': already_refreshing}\n\n        if not already_refreshing and (is_stale or not cached):"))
    else:
        new_lines.append(line)

with open("backend/api/routes/outreach.py", "w") as f:
    f.writelines(new_lines)

print("Fixed hotpaths.")
