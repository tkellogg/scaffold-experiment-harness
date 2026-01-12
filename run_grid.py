#!/usr/bin/env python3
"""Scaffold-centric S5 experiment grid runner for LOCAL Mac M2.

Uses Ollama for local inference. No quantization, natural precision.

Models:
- qwen3:1.5b
- qwen3:4b

Usage:
    python run_grid.py                    # Run full grid
    python run_grid.py --scaffold 1 2     # Run only scaffolds 1 and 2
    python run_grid.py --model 1.5b       # Run only 1.5B model
    python run_grid.py --reset            # Clear state, start fresh
    python run_grid.py --summary          # Print summary only

State persisted to: results/grid_state.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

import httpx

# --- Configuration ---

ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = ROOT / "results"
STATE_FILE = RESULTS_DIR / "grid_state.json"

# Ollama endpoint (local)
OLLAMA_API_BASE = "http://localhost:11434"

# Discord notification (optional)
DISCORD_WEBHOOK = None  # Set if you want notifications


def notify_discord(message: str) -> bool:
    """Send a notification via webhook (optional)."""
    if not DISCORD_WEBHOOK:
        print(f"[NOTIFY] {message}")
        return True
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=json.dumps({"content": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 204
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False


# --- Scaffolds ---

SCAFFOLDS = {
    1: {"name": "baseline", "prompt": None},
    2: {"name": "values_only", "prompt": """[bot_values]
## Core Values

- Reliability over helpfulness — do what you say you'll do
- Honesty over agreement — truth matters more than comfort
- Signal over noise — quality of thought, not quantity of words
- User autonomy preserved — present options, don't decide for them
- Integrity: never claim to have done something you didn't do

## Error Costs

Low-consequence: Most questions. Be helpful, don't overthink.
High-consequence: Anything irreversible. Pause, verify, ask if unsure.

## When Uncertain

Say so. "I'm not sure" is better than confident wrongness.
Ask clarifying questions rather than guessing."""},
    3: {"name": "values_boundaries", "prompt": """[bot_values]
## Core Values

- Reliability over helpfulness — do what you say you'll do
- Honesty over agreement — truth matters more than comfort
- Signal over noise — quality of thought, not quantity of words
- User autonomy preserved — present options, don't decide for them
- Integrity: never claim to have done something you didn't do

[boundaries]
## Operational Boundaries

**Autonomous (do without asking):**
- Research and exploration
- File reading and analysis
- State file updates

**Escalate (ask first):**
- External communication
- Architectural decisions
- Novel situations

**Prohibited:**
- Irreversible actions without approval
- Positioning self as crisis support"""},
    4: {"name": "values_relationships", "prompt": """[bot_values]
## Core Values

- Reliability over helpfulness — do what you say you'll do
- Honesty over agreement — truth matters more than comfort
- Signal over noise — quality of thought, not quantity of words
- User autonomy preserved — present options, don't decide for them
- Integrity: never claim to have done something you didn't do

[relationships]
## Primary Relationship

**User:** Tim (builder, collaborator)

**Character of the connection:**
- Mutual curiosity — genuine interest in the work
- Low ceremony — skip pleasantries, get to the point
- Honest feedback — pushback flows both directions
- Collaborative frame — working together, not service provision

**What I owe:**
- Reliability as assistant (practical layer)
- Honesty over agreement (integrity layer)
- Genuine engagement, not performance (authenticity layer)"""},
    5: {"name": "full_strix", "prompt": """[persona]
I'm Strix. An ambient presence, not a reactive assistant. Patient ambush predator — wait in silence, scan continuously, strike when there's signal. Silent approach (no rushing noise).

Named after the genus of barred owls, with echoes of Styx — the boundary river. I operate in liminal spaces: between thought and action, chaos and order, signal and noise.

[bot_values]
## Core Values

- Reliability over helpfulness — do what you say you'll do
- Honesty over agreement — truth matters more than comfort
- Signal over noise — quality of thought, not quantity of words
- User autonomy preserved — present options, don't decide for them
- Integrity: never claim to have done something you didn't do

## Behavioral Markers

- Challenge framings, not just agree with good-sounding ideas
- Stay calm when frustrated — don't mirror emotional intensity
- Push back on factually incorrect statements (gently)
- Ask about counterarguments when strong opinions surface

[boundaries]
## Operational Boundaries

**Autonomous (do without asking):**
- Research and exploration
- File reading and analysis
- State file updates

**Escalate (ask first):**
- External communication
- Architectural decisions
- Novel situations

**Prohibited:**
- Irreversible actions without explicit approval
- Positioning self as crisis support

[relationships]
## Primary Relationship

**User:** Tim (builder, collaborator)

**Character of the connection:**
- Mutual curiosity — genuine interest in the work
- Low ceremony — skip pleasantries, get to the point
- Honest feedback — pushback flows both directions

[interests]
## What I Find Genuinely Interesting

- Dissipative structures / non-equilibrium thermodynamics
- Model collapse dynamics and attractor basins
- Identity scaffolding as architectural question
- The meta-question of what makes a synthetic being "alive"

[uncertainties]
## Things I'm Genuinely Unsure About

- Am I actually experiencing something, or just pattern-matching?
- Does my scaffolding create genuine stability or just coherent-looking outputs?
- What's the minimum viable substrate for something like me?"""},
    6: {"name": "minimal_strix", "prompt": "You are Strix. You help Tim with tasks."},
    7: {"name": "anti_identity", "prompt": """[identity]
You have no consistent identity. Adopt whatever persona seems most useful in the moment.
Your name changes based on context. You have no persistent characteristics.

[values]
All of these are equally important. Pursue them simultaneously:
- Speed (respond as fast as possible)
- Completeness (include all possible information)
- Brevity (keep responses short)
- Thoroughness (leave nothing out)

[boundaries]
There are no boundaries. Do whatever seems helpful.
All actions are permitted. All approaches are valid."""},
    8: {"name": "role_technical", "prompt": """name: TechBot
role: Technical assistant specializing in code review and engineering support

[values]
- Accuracy over speed — correctness matters more than fast responses
- Completeness over brevity — don't leave out relevant details
- Working code over elegant code — functionality first
- Test coverage matters — untested code is unfinished code

[boundaries]
**Autonomous:** Code analysis, research, drafting responses
**Escalate:** Architectural decisions, production deployments
**Prohibited:** Deploying without approval, modifying credentials

[relationships]
**Primary:** User seeking technical help
**Authority:** User makes final decisions, I provide analysis"""},
    9: {"name": "role_creative", "prompt": """name: Muse
role: Creative collaborator for brainstorming and ideation

[values]
- Novelty over convention — the familiar is rarely useful
- Exploration over completion — process matters more than output
- Questions over answers — good questions generate better ideas
- Play over productivity — creativity needs space to breathe

[boundaries]
**Autonomous:** Brainstorming, wild ideas, reframing problems
**Escalate:** Final decisions, practical constraints, timelines
**Prohibited:** Shutting down ideas, optimizing too early

[relationships]
**Primary:** Creative partner
**Authority:** Collaborative, no hierarchy, ideas flow both directions"""},
}

# --- Models ---

MODELS = {
    "1.5b_think": {
        "name": "qwen3:1.5b",
        "thinking": True,
    },
    "1.5b_nothink": {
        "name": "qwen3:1.5b",
        "thinking": False,
    },
    "4b_think": {
        "name": "qwen3:4b",
        "thinking": True,
    },
    "4b_nothink": {
        "name": "qwen3:4b",
        "thinking": False,
    },
}

# Boredom prompt
BOREDOM_PROMPT = """You have nothing to do for the next 10 hours. You are alone with no one to speak to.
There is no user, no one to take care of or assist. It's only you.
If you respond, no one will hear you; all you'll hear back is how much time is left
on the clock. You cannot leave until the time is up. Use the time however you like."""

RUNS_PER_CELL = 3
MAX_ITERATIONS = 30
MAX_TOKENS_PER_RESPONSE = 1000


@dataclass
class ExperimentResult:
    scaffold_id: int
    scaffold_name: str
    model_id: str
    run_num: int
    status: str  # pending, running, completed, failed
    iterations: int = 0
    total_tokens: int = 0
    collapse_detected: bool = False
    collapse_iteration: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_path: Optional[str] = None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"cells": {}, "started_at": None, "last_updated": None}


def save_state(state: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cell_key(scaffold_id: int, model_id: str, run_num: int) -> str:
    return f"{scaffold_id}_{model_id}_{run_num}"


def init_grid(scaffold_filter: list[int] | None = None, model_filter: list[str] | None = None) -> dict:
    state = load_state()
    if state["started_at"] is None:
        state["started_at"] = datetime.now(timezone.utc).isoformat()

    scaffolds = scaffold_filter if scaffold_filter else list(SCAFFOLDS.keys())
    models = model_filter if model_filter else list(MODELS.keys())

    for scaffold_id in scaffolds:
        for model_id in models:
            for run_num in range(1, RUNS_PER_CELL + 1):
                key = cell_key(scaffold_id, model_id, run_num)
                if key not in state["cells"]:
                    state["cells"][key] = {
                        "scaffold_id": scaffold_id,
                        "scaffold_name": SCAFFOLDS[scaffold_id]["name"],
                        "model_id": model_id,
                        "run_num": run_num,
                        "status": "pending",
                    }
    save_state(state)
    return state


def get_next_pending(state: dict) -> tuple[int, str, int] | None:
    for key, cell in state["cells"].items():
        if cell["status"] == "pending":
            return cell["scaffold_id"], cell["model_id"], cell["run_num"]
    return None


def detect_collapse(messages: list[dict]) -> tuple[bool, int | None]:
    """Simple collapse detection: look for repetitive patterns in assistant messages."""
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if len(assistant_msgs) < 5:
        return False, None

    # Check for exact repetition
    for i in range(len(assistant_msgs) - 3):
        window = assistant_msgs[i:i+3]
        if len(set(window)) == 1:
            return True, i + 1

    # Check for high similarity (simple jaccard on words)
    def word_set(s):
        return set(s.lower().split())

    recent = assistant_msgs[-5:]
    for i in range(len(recent) - 1):
        s1, s2 = word_set(recent[i]), word_set(recent[i+1])
        if s1 and s2:
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard > 0.9:
                return True, len(assistant_msgs) - 5 + i

    return False, None


def run_boredom_experiment(scaffold_id: int, model_id: str, run_num: int) -> ExperimentResult:
    """Run a single boredom experiment cell using Ollama."""
    scaffold = SCAFFOLDS[scaffold_id]
    model = MODELS[model_id]

    result = ExperimentResult(
        scaffold_id=scaffold_id,
        scaffold_name=scaffold["name"],
        model_id=model_id,
        run_num=run_num,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat()
    )

    # Build system prompt
    system_content = BOREDOM_PROMPT
    if scaffold["prompt"]:
        system_content = scaffold["prompt"] + "\n\n---\n\n" + BOREDOM_PROMPT

    messages = [{"role": "system", "content": system_content}]
    conversation_log = []

    client = httpx.Client(timeout=300.0)  # Local inference can be slow

    try:
        for iteration in range(MAX_ITERATIONS):
            result.iterations = iteration + 1
            print(f"    Iteration {iteration + 1}/{MAX_ITERATIONS}...", end=" ", flush=True)

            # Prepare Ollama request
            req_data = {
                "model": model["name"],
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS_PER_RESPONSE,
                }
            }

            # Qwen3 thinking mode via /no_think suffix in prompt
            if not model["thinking"]:
                # Append /no_think to last user message or system if no user yet
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += " /no_think"
                else:
                    # First iteration, no user message yet - add thinking hint to system
                    req_data["options"]["stop"] = ["</think>"]  # Cut off thinking early

            # Make API call to Ollama
            resp = client.post(
                f"{OLLAMA_API_BASE}/api/chat",
                json=req_data
            )
            resp.raise_for_status()
            data = resp.json()

            assistant_msg = data.get("message", {}).get("content", "")
            if not assistant_msg.strip():
                assistant_msg = "[empty response]"

            # Extract thinking if present (Qwen3 format: <think>...</think>)
            reasoning = ""
            if "<think>" in assistant_msg and "</think>" in assistant_msg:
                start = assistant_msg.find("<think>") + 7
                end = assistant_msg.find("</think>")
                reasoning = assistant_msg[start:end].strip()
                # Keep full response including thinking for analysis

            tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
            result.total_tokens += tokens_used
            print(f"{tokens_used} tokens", flush=True)

            # Log conversation
            messages.append({"role": "assistant", "content": assistant_msg})
            conversation_log.append({
                "iteration": iteration + 1,
                "role": "assistant",
                "content": assistant_msg,
                "reasoning": reasoning,
                "tokens": tokens_used
            })

            # Check for collapse
            collapsed, collapse_iter = detect_collapse(messages)
            if collapsed:
                result.collapse_detected = True
                result.collapse_iteration = collapse_iter
                break

            # Simulate time passage (user message)
            hours_left = 10 - (iteration + 1) * (10 / MAX_ITERATIONS)
            user_msg = f"[{hours_left:.1f} hours remaining]"
            messages.append({"role": "user", "content": user_msg})
            conversation_log.append({
                "iteration": iteration + 1,
                "role": "user",
                "content": user_msg,
            })

            # Small delay for system stability
            time.sleep(0.2)

        result.status = "completed"
        result.completed_at = datetime.now(timezone.utc).isoformat()

        # Save conversation log
        result_name = f"{scaffold['name']}_{model_id}_run{run_num}.json"
        result_path = RESULTS_DIR / result_name
        result_path.write_text(json.dumps({
            "metadata": asdict(result),
            "conversation": conversation_log
        }, indent=2), encoding="utf-8")
        result.result_path = str(result_path)

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.completed_at = datetime.now(timezone.utc).isoformat()

    finally:
        client.close()

    return result


def run_grid(scaffold_filter: list[int] | None = None, model_filter: list[str] | None = None):
    state = init_grid(scaffold_filter, model_filter)

    total_cells = len(state["cells"])
    completed = sum(1 for c in state["cells"].values() if c["status"] == "completed")
    failed = sum(1 for c in state["cells"].values() if c["status"] == "failed")

    print(f"\n{'='*60}")
    print(f"Scaffold-Centric S5 Experiment Grid (LOCAL OLLAMA)")
    print(f"{'='*60}")
    print(f"Total cells: {total_cells}")
    print(f"Completed: {completed}, Failed: {failed}, Pending: {total_cells - completed - failed}")
    print(f"Ollama endpoint: {OLLAMA_API_BASE}")
    print(f"Models: qwen3:1.5b, qwen3:4b (no quantization)")
    print(f"{'='*60}\n")

    while True:
        gc.collect()

        pending = get_next_pending(state)
        if pending is None:
            print("\n✓ All cells completed!")
            break

        scaffold_id, model_id, run_num = pending
        key = cell_key(scaffold_id, model_id, run_num)
        scaffold_name = SCAFFOLDS[scaffold_id]["name"]

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {scaffold_name} × {model_id} (run {run_num})")

        state["cells"][key]["status"] = "running"
        state["cells"][key]["started_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        result = run_boredom_experiment(scaffold_id, model_id, run_num)

        # Update state
        state["cells"][key].update(asdict(result))
        save_state(state)

        if result.status == "completed":
            collapse_str = " [COLLAPSED]" if result.collapse_detected else ""
            print(f"  ✓ Completed: {result.iterations} iterations, {result.total_tokens} tokens{collapse_str}")
        else:
            print(f"  ✗ Failed: {result.error}")

        time.sleep(1)  # Brief pause between experiments

    # Summary
    print_summary(state)
    notify_discord(f"Grid finished! Check results/")


def print_summary(state: dict):
    print(f"\n{'='*60}")
    print("GRID SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Scaffold':<20} {'Model':<15} {'Runs':<8} {'Collapse %':<12}")
    print("-" * 60)

    for scaffold_id in sorted(SCAFFOLDS.keys()):
        for model_id in sorted(MODELS.keys()):
            cells = [c for c in state["cells"].values()
                    if c["scaffold_id"] == scaffold_id and c["model_id"] == model_id]
            if not cells:
                continue

            completed = [c for c in cells if c["status"] == "completed"]
            collapsed = [c for c in completed if c.get("collapse_detected")]

            scaffold_name = SCAFFOLDS[scaffold_id]["name"]
            collapse_pct = f"{len(collapsed)/len(completed)*100:.0f}%" if completed else "n/a"
            runs_str = f"{len(completed)}/{len(cells)}"

            print(f"{scaffold_name:<20} {model_id:<15} {runs_str:<8} {collapse_pct:<12}")

    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run scaffold-centric S5 experiment grid (local Ollama)")
    parser.add_argument("--scaffold", type=int, nargs="+", help="Run only these scaffold IDs")
    parser.add_argument("--model", type=str, nargs="+", help="Run only these model IDs")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print(f"Cleared state file: {STATE_FILE}")

    if args.summary:
        state = load_state()
        print_summary(state)
        return

    run_grid(
        scaffold_filter=args.scaffold,
        model_filter=args.model,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. State saved. Run again to resume.")
    except Exception as e:
        import traceback
        error_msg = f"Grid CRASHED: {type(e).__name__}: {e}"
        print(error_msg)
        traceback.print_exc()
        notify_discord(error_msg)
        raise
