#!/usr/bin/env python3
"""
Muninn Auto-Ingestion — Automatically ingest OpenClaw session transcripts.

Exports recent session trajectories, extracts user/assistant messages,
and ingests them into Muninn Local as raw sessions.

Tracks which sessions have already been ingested to avoid duplicates.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add muninn-local to path
MUNINN_DIR = os.path.expanduser("~/.openclaw/workspace/muninn-local")
sys.path.insert(0, MUNINN_DIR)

from muninn import ingest_session, get_db, init_db

# State file to track ingested sessions
STATE_FILE = os.path.join(MUNINN_DIR, ".ingested-sessions.json")

# Only ingest sessions updated within this many minutes
RECENT_WINDOW_MINUTES = 360  # 6 hours

# Minimum number of user messages to bother ingesting
MIN_MESSAGES = 2


def load_state():
    """Load the set of already-ingested session keys."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"ingested": [], "last_run": None}


def save_state(state):
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_recent_sessions():
    """Get recent OpenClaw sessions as JSON."""
    result = subprocess.run(
        ["openclaw", "sessions", "--json", "--limit", "20"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        print(f"✗ Failed to list sessions: {result.stderr}")
        return []

    data = json.loads(result.stdout)
    sessions = data.get("sessions", [])

    # Filter to recent sessions
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=RECENT_WINDOW_MINUTES)
    recent = []
    for s in sessions:
        updated = datetime.fromtimestamp(s["updatedAt"] / 1000, tz=timezone.utc)
        if updated >= cutoff:
            recent.append(s)

    return recent


def export_session_transcript(session_key):
    """Export a session trajectory and return the user/assistant messages."""
    result = subprocess.run(
        ["openclaw", "sessions", "export-trajectory",
         "--session-key", session_key, "--json"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"  ✗ Export failed for {session_key}: {result.stderr[:100]}")
        return None

    export_info = json.loads(result.stdout)
    output_dir = export_info["outputDir"]

    # Read events.jsonl and extract user/assistant messages
    events_file = os.path.join(output_dir, "events.jsonl")
    if not os.path.exists(events_file):
        print(f"  ✗ No events.jsonl in {output_dir}")
        return None

    messages = []
    speakers = set()
    with open(events_file) as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") in ("user.message", "assistant.message"):
                msg_data = event.get("data", {}).get("message", {})
                role = msg_data.get("role", "")
                content = msg_data.get("content", "")

                # Extract text from content (can be string or list of parts)
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text += part.get("text", "")
                        elif isinstance(part, str):
                            text += part

                if text.strip():
                    # Skip system/heartbeat noise
                    if text.strip() in ("HEARTBEAT_OK", "NO_REPLY"):
                        continue
                    if text.strip().startswith("[[") and text.strip().endswith("]]"):
                        continue

                    messages.append({
                        "role": role,
                        "text": text.strip(),
                        "timestamp": event.get("ts", "")
                    })
                    speakers.add(role)

    return messages, list(speakers), output_dir


def build_conversation_text(messages):
    """Build a readable conversation transcript from messages."""
    lines = []
    for msg in messages:
        speaker = "Alex" if msg["role"] == "user" else "Leo"
        lines.append(f"[{speaker}] ({msg['timestamp']}): {msg['text']}")
    return "\n\n".join(lines)


def auto_ingest():
    """Main auto-ingestion loop."""
    state = load_state()
    ingested_keys = set(state.get("ingested", []))

    sessions = get_recent_sessions()
    if not sessions:
        print("✓ No recent sessions to ingest")
        return

    print(f"→ Found {len(sessions)} recent sessions")

    newly_ingested = 0
    for session in sessions:
        session_key = session["key"]

        if session_key in ingested_keys:
            continue

        # Skip sessions with very few tokens (probably just system messages)
        if session.get("inputTokens", 0) < 100:
            continue

        print(f"  → Exporting {session_key}...")

        result = export_session_transcript(session_key)
        if result is None:
            continue

        messages, speakers, export_dir = result

        if len(messages) < MIN_MESSAGES:
            print(f"  ✗ Only {len(messages)} messages, skipping")
            ingested_keys.add(session_key)
            continue

        # Build conversation text
        conversation = build_conversation_text(messages)

        # Determine source and channel from session key
        parts = session_key.split(":")
        source = parts[2] if len(parts) > 2 else "openclaw"  # discord, telegram, etc.
        channel = parts[3] if len(parts) > 3 else None  # channel, direct, etc.

        # Map speaker roles to names
        named_speakers = []
        for s in speakers:
            if s == "user":
                named_speakers.append("Alex")
            elif s == "assistant":
                named_speakers.append("Leo")
            else:
                named_speakers.append(s)

        # Get session date from updatedAt
        session_date = datetime.fromtimestamp(
            session["updatedAt"] / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        # Ingest into Muninn
        session_id = ingest_session(
            content=conversation,
            source=source,
            speakers=named_speakers,
            channel=channel,
            session_date=session_date
        )
        print(f"  ✓ Ingested as {session_id} ({len(messages)} messages)")

        ingested_keys.add(session_key)
        newly_ingested += 1

        # Clean up the export directory to save disk space
        try:
            import shutil
            shutil.rmtree(export_dir, ignore_errors=True)
        except Exception:
            pass

    # Update state
    state["ingested"] = list(ingested_keys)
    save_state(state)

    if newly_ingested > 0:
        print(f"✓ Ingested {newly_ingested} new sessions")
        print(f"→ Running fact extraction on unprocessed sessions...")
        # Auto-extract facts from newly ingested sessions
        from muninn import extract_all_unprocessed
        extract_all_unprocessed()
    else:
        print("✓ No new sessions to ingest")


if __name__ == "__main__":
    auto_ingest()