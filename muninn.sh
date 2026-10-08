#!/bin/bash
# Muninn Local — wrapper for easy CLI usage
# Usage: ./muninn.sh [command] [args]
# Special: ./muninn.sh auto-ingest  → auto-ingest recent OpenClaw sessions

cd "$(dirname "$0")"

# Ensure openclaw CLI is reachable (crontab has minimal PATH; agent sessions get a tmp shim)
export PATH="/home/homelab/.nvm/versions/node/v24.21.0/bin:$PATH"

# Load Ollama Cloud API key from OpenClaw auth store
# Primary: shared state DB config_machine_state.authProfiles.store (OpenClaw migrated auth here ~Sep 2026)
# Fallback: legacy agent DB auth_profile_store
export OLLAMA_API_KEY=$(python3 -c "
import sqlite3, json
try:
    db = sqlite3.connect('/home/homelab/.openclaw/state/openclaw.sqlite')
    row = db.execute(\"SELECT value_json FROM config_machine_state WHERE state_key='authProfiles.store'\").fetchone()
    if row:
        data = json.loads(row[0])
        key = data.get('profiles', {}).get('ollama-cloud:default', {}).get('key', '')
        if key:
            print(key)
    db.close()
except Exception:
    pass
" 2>/dev/null)
if [ -z "$OLLAMA_API_KEY" ]; then
export OLLAMA_API_KEY=$(python3 -c "
import sqlite3, json
db = sqlite3.connect('/home/homelab/.openclaw/agents/main/agent/openclaw-agent.sqlite')
row = db.execute(\"SELECT store_json FROM auth_profile_store WHERE store_key='primary'\").fetchone()
data = json.loads(row[0])
print(data['profiles']['ollama-cloud:default']['key'])
db.close()
" 2>/dev/null)
fi

if [ "$1" = "auto-ingest" ]; then
    python3 auto_ingest.py
elif [ "$1" = "librarian" ]; then
    shift
    python3 librarian.py "$@"
else
    python3 muninn.py "$@"
fi