#!/usr/bin/env python
"""Patch config.py to add demo mode support for assessments auth."""

with open('C:\\Users\\HP\\amd-hackathon-2026\\apps\\api\\telltale_api\\core\\config.py') as f:
    content = f.read()

# 1. Add demo_mode setting after ai_provider
old_ai_provider = '    ai_provider: str = "template"'
new_ai_provider = '''    ai_provider: str = "template"

    # Demo mode for local development without login.
    # When TELLTALE_DEMO_MODE=1, all assessment/AI output routes use a fixed
    # local owner "local-demo-user" and a fixed demo token, bypassing the
    # normal TELLTALE_API_TOKENS authentication requirement. When disabled,
    # the normal auth is enforced strictly.
    demo_mode: bool = False'''

if old_ai_provider in content:
    content = content.replace(old_ai_provider, new_ai_provider)
    print("Added demo_mode setting")
else:
    print("Could not find ai_provider line")

# 2. Add demo token default to api_tokens
old_api_tokens = '    api_tokens: str = ""'
new_api_tokens = '''    api_tokens: str = ""\n\n    # When TELLTALE_DEMO_MODE=1 and no explicit api_tokens are set,\n    # use a fixed demo token and owner. The normal auth behavior is preserved\n    # when demo_mode is False or explicit tokens are configured.'''

if old_api_tokens in content:
    content = content.replace(old_api_tokens, new_api_tokens)
    print("Added demo mode comment to api_tokens")
else:
    print("Could not find api_tokens line")

with open('C:\\Users\\HP\\amd-hackathon-2026\\apps\\api\\telltale_api\\core\\config.py', 'w') as f:
    f.write(content)

print("Config patch complete")