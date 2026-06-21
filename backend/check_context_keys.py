from backend.services.ai_columns import build_candidate_context
from backend.api.routes.ai_columns import _fetch_profile

profile = _fetch_profile(7653)
ctx = build_candidate_context(profile)
print("Keys in context:")
for k in sorted(ctx.keys()):
    if "inkedin" in k.lower():
        print(k)
