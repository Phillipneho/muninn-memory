#!/bin/bash
# muninn_recall.sh — Pre-answer recall hook for OpenClaw
#
# Usage: ./muninn_recall.sh "user's question here"
#
# Outputs a JSON context bundle that can be injected into the conversation
# as pre-answer memory context. This runs the full hybrid retrieval pipeline:
#   1. Intent-aware retrieval planning (LLM generates a plan)
#   2. Hybrid search (semantic + lexical/FTS5 + structured/PDS)
#   3. Reciprocal Rank Fusion to merge results
#   4. Compact context bundle assembly within token budget
#
# Called from OpenClaw's memory pipeline before answering user questions.

cd "$(dirname "$0")"

# Load Ollama Cloud API key from OpenClaw auth store
export OLLAMA_API_KEY=$(python3 -c "
import sqlite3, json
db = sqlite3.connect('/home/homelab/.openclaw/agents/main/agent/openclaw-agent.sqlite')
row = db.execute(\"SELECT store_json FROM auth_profile_store WHERE store_key='primary'\").fetchone()
data = json.loads(row[0])
print(data['profiles']['ollama-cloud:default']['key'])
db.close()
" 2>/dev/null)

QUERY="$1"

if [ -z "$QUERY" ]; then
    echo '{"error": "No query provided. Usage: ./muninn_recall.sh \"user question\""}' 
    exit 1
fi

# Run pre-answer recall and output JSON
python3 muninn.py recall-context "$QUERY" 2>/dev/null