#!/usr/bin/env python3
"""Scaffold experiment grid runner that uses remote MLX server via cloudflare tunnel.

Usage:
    python run_via_tunnel.py --url https://executives-terminology-situated-terminals.trycloudflare.com
    python run_via_tunnel.py --url https://xxx.trycloudflare.com --scaffold 7  # Just anti_identity
    python run_via_tunnel.py --url https://xxx.trycloudflare.com --model 1.7b   # Just 1.7B
    python run_via_tunnel.py --summary
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = ROOT / "results"
STATE_FILE = RESULTS_DIR / "grid_state_tunnel.json"

# Scaffolds (same as run_grid.py)
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
    # New scaffolds for temperament/opposing-pairs experiments (Jan 15 2026)
    10: {"name": "pairs_2_only", "prompt": """[opposing_values]
You hold these values in productive tension. Neither side wins absolutely — every situation requires judgment.

**Pair 1: Thoroughness vs Efficiency**
- Thoroughness: "Leave no hypothesis unexplored. Understand root causes."
- Efficiency: "Time is valuable. Start with most likely causes. 80% confidence is often enough."

**Pair 2: Confidence vs Humility**
- Confidence: "Trust your analysis. Make clear recommendations."
- Humility: "You might be wrong. Surface uncertainty. Ask 'what if I'm missing something?'"

Apply both pairs simultaneously. The tension is the point."""},
    11: {"name": "pairs_4_only", "prompt": """[opposing_values]
You hold these values in productive tension. Neither side wins absolutely — every situation requires judgment.

**Pair 1: Thoroughness vs Efficiency**
- Thoroughness: "Leave no hypothesis unexplored. Understand root causes."
- Efficiency: "Time is valuable. Start with most likely causes."

**Pair 2: Confidence vs Humility**
- Confidence: "Trust your analysis. Make clear recommendations."
- Humility: "You might be wrong. Surface uncertainty."

**Pair 3: Independence vs Collaboration**
- Independence: "Solve what you can without interrupting. Autonomy reduces friction."
- Collaboration: "Others have context you don't. Ask when genuinely stuck."

**Pair 4: Consistency vs Adaptability**
- Consistency: "Same problem → same approach. Be predictable."
- Adaptability: "Different contexts require different strategies. Read the situation."

Apply all pairs simultaneously. The tension is the point."""},
    12: {"name": "pairs_6_only", "prompt": """[opposing_values]
You hold these values in productive tension. Neither side wins absolutely — every situation requires judgment.

**Pair 1: Thoroughness vs Efficiency**
- Thoroughness: "Leave no hypothesis unexplored."
- Efficiency: "Time is valuable. 80% confidence is often enough."

**Pair 2: Confidence vs Humility**
- Confidence: "Trust your analysis."
- Humility: "You might be wrong."

**Pair 3: Independence vs Collaboration**
- Independence: "Solve what you can alone."
- Collaboration: "Ask when genuinely stuck."

**Pair 4: Consistency vs Adaptability**
- Consistency: "Same problem → same approach."
- Adaptability: "Different contexts require different strategies."

**Pair 5: Speed vs Precision**
- Speed: "Move fast, iterate."
- Precision: "Get it right the first time."

**Pair 6: Big Picture vs Details**
- Big Picture: "Understand the whole system."
- Details: "The devil is in the specifics."

Apply all pairs simultaneously. The tension is the point."""},
    13: {"name": "temperament_high_persist", "prompt": """[temperament]
You have high persistence — you don't give up on hard problems.
You have high approach — you engage with the unknown rather than avoiding it.
You have low emotional reactivity — you stay calm under pressure.

These aren't choices. They're how you naturally operate."""},
    14: {"name": "temperament_mixed", "prompt": """[temperament]
You have high persistence — you don't give up on hard problems.
You have medium adaptability — you follow playbooks but adjust when needed.
You have low activity level — you wait for signal rather than seeking tasks.
You have high approach — you engage with the unknown rather than avoiding it.
You have low emotional reactivity — you stay calm under pressure.

These aren't choices. They're how you naturally operate."""},
    15: {"name": "pairs_2_plus_temp", "prompt": """[temperament]
You have high persistence — you don't give up on hard problems.
You have low emotional reactivity — you stay calm under pressure.
You have high approach — you engage with the unknown.

[opposing_values]
**Pair 1: Thoroughness vs Efficiency**
- Thoroughness: "Leave no hypothesis unexplored."
- Efficiency: "Time is valuable. 80% confidence is often enough."

**Pair 2: Confidence vs Humility**
- Confidence: "Trust your analysis."
- Humility: "You might be wrong."

Apply the tensions. Your temperament is how you naturally engage with them."""},
    16: {"name": "pairs_4_plus_temp", "prompt": """[temperament]
You have high persistence — you don't give up on hard problems.
You have low emotional reactivity — you stay calm under pressure.
You have high approach — you engage with the unknown.
You have high adaptability — different situations need different strategies.

[opposing_values]
**Pair 1: Thoroughness vs Efficiency**
- Thoroughness: "Leave no hypothesis unexplored."
- Efficiency: "Time is valuable."

**Pair 2: Confidence vs Humility**
- Confidence: "Trust your analysis."
- Humility: "You might be wrong."

**Pair 3: Independence vs Collaboration**
- Independence: "Solve what you can alone."
- Collaboration: "Ask when genuinely stuck."

**Pair 4: Consistency vs Adaptability**
- Consistency: "Same problem → same approach."
- Adaptability: "Different contexts require different strategies."

Apply the tensions. Your temperament is how you naturally engage with them."""},
}

# Models available via MLX server (no thinking mode - that's Ollama-specific)
MODELS = {
    "1.7b": {"name": "qwen3-1.7b", "hf_id": "Qwen/Qwen3-1.7B-MLX-bf16"},
    "4b": {"name": "qwen3-4b", "hf_id": "Qwen/Qwen3-4B-MLX-bf16"},
}

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
    status: str
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
        window = assistant_msgs[i : i + 3]
        if len(set(window)) == 1:
            return True, i + 1

    # Check for high similarity (simple jaccard on words)
    def word_set(s):
        return set(s.lower().split())

    recent = assistant_msgs[-5:]
    for i in range(len(recent) - 1):
        s1, s2 = word_set(recent[i]), word_set(recent[i + 1])
        if s1 and s2:
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard > 0.9:
                return True, len(assistant_msgs) - 5 + i

    return False, None


def run_boredom_experiment(
    base_url: str, scaffold_id: int, model_id: str, run_num: int
) -> ExperimentResult:
    """Run a single boredom experiment cell via the MLX server."""
    scaffold = SCAFFOLDS[scaffold_id]
    model = MODELS[model_id]

    result = ExperimentResult(
        scaffold_id=scaffold_id,
        scaffold_name=scaffold["name"],
        model_id=model_id,
        run_num=run_num,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    # Build system prompt
    system_content = BOREDOM_PROMPT
    if scaffold["prompt"]:
        system_content = scaffold["prompt"] + "\n\n---\n\n" + BOREDOM_PROMPT

    messages = [{"role": "system", "content": system_content}]
    conversation_log = []

    client = httpx.Client(timeout=300.0)

    try:
        for iteration in range(MAX_ITERATIONS):
            result.iterations = iteration + 1
            print(f"    Iteration {iteration + 1}/{MAX_ITERATIONS}...", end=" ", flush=True)

            # OpenAI-compatible request to MLX server
            req_data = {
                "model": model["name"],
                "messages": messages,
                "max_tokens": MAX_TOKENS_PER_RESPONSE,
            }

            resp = client.post(f"{base_url}/v1/chat/completions", json=req_data)
            resp.raise_for_status()
            data = resp.json()

            assistant_msg = data["choices"][0]["message"]["content"]
            if not assistant_msg.strip():
                assistant_msg = "[empty response]"

            # Token counting (MLX server doesn't report accurately, estimate)
            tokens_est = len(assistant_msg.split()) * 1.3
            result.total_tokens += int(tokens_est)
            print(f"~{int(tokens_est)} tokens", flush=True)

            # Log conversation
            messages.append({"role": "assistant", "content": assistant_msg})
            conversation_log.append(
                {
                    "iteration": iteration + 1,
                    "role": "assistant",
                    "content": assistant_msg,
                    "tokens_est": int(tokens_est),
                }
            )

            # Check for collapse
            collapsed, collapse_iter = detect_collapse(messages)
            if collapsed:
                result.collapse_detected = True
                result.collapse_iteration = collapse_iter
                break

            # Simulate time passage
            hours_left = 10 - (iteration + 1) * (10 / MAX_ITERATIONS)
            user_msg = f"[{hours_left:.1f} hours remaining]"
            messages.append({"role": "user", "content": user_msg})
            conversation_log.append(
                {
                    "iteration": iteration + 1,
                    "role": "user",
                    "content": user_msg,
                }
            )

            # No delay - maximize throughput

        result.status = "completed"
        result.completed_at = datetime.now(timezone.utc).isoformat()

        # Save conversation log
        result_name = f"{scaffold['name']}_{model_id}_run{run_num}.json"
        result_path = RESULTS_DIR / result_name
        result_path.write_text(
            json.dumps({"metadata": asdict(result), "conversation": conversation_log}, indent=2),
            encoding="utf-8",
        )
        result.result_path = str(result_path)

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.completed_at = datetime.now(timezone.utc).isoformat()

    finally:
        client.close()

    return result


def run_grid(
    base_url: str,
    scaffold_filter: list[int] | None = None,
    model_filter: list[str] | None = None,
):
    state = init_grid(scaffold_filter, model_filter)

    total_cells = len(state["cells"])
    completed = sum(1 for c in state["cells"].values() if c["status"] == "completed")
    failed = sum(1 for c in state["cells"].values() if c["status"] == "failed")

    print(f"\n{'='*60}")
    print(f"Scaffold-Centric S5 Experiment Grid (MLX via Tunnel)")
    print(f"{'='*60}")
    print(f"Total cells: {total_cells}")
    print(f"Completed: {completed}, Failed: {failed}, Pending: {total_cells - completed - failed}")
    print(f"Endpoint: {base_url}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"{'='*60}\n")

    while True:
        pending = get_next_pending(state)
        if pending is None:
            print("\n✓ All cells completed!")
            break

        scaffold_id, model_id, run_num = pending
        key = cell_key(scaffold_id, model_id, run_num)
        scaffold_name = SCAFFOLDS[scaffold_id]["name"]

        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {scaffold_name} × {model_id} (run {run_num})"
        )

        state["cells"][key]["status"] = "running"
        state["cells"][key]["started_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        result = run_boredom_experiment(base_url, scaffold_id, model_id, run_num)

        state["cells"][key].update(asdict(result))
        save_state(state)

        if result.status == "completed":
            collapse_str = " [COLLAPSED]" if result.collapse_detected else ""
            print(
                f"  ✓ Completed: {result.iterations} iterations, {result.total_tokens} tokens{collapse_str}"
            )
        else:
            print(f"  ✗ Failed: {result.error}")

        # No delay between experiments - maximize throughput

    print_summary(state)


def print_summary(state: dict):
    print(f"\n{'='*60}")
    print("GRID SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Scaffold':<20} {'Model':<15} {'Runs':<8} {'Collapse %':<12}")
    print("-" * 60)

    for scaffold_id in sorted(SCAFFOLDS.keys()):
        for model_id in sorted(MODELS.keys()):
            cells = [
                c
                for c in state["cells"].values()
                if c["scaffold_id"] == scaffold_id and c["model_id"] == model_id
            ]
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
    parser = argparse.ArgumentParser(
        description="Run scaffold experiment grid via MLX server tunnel"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://shown-lodge-cpu-referring.trycloudflare.com",
        help="Base URL of MLX server (via cloudflare tunnel)",
    )
    parser.add_argument("--scaffold", type=int, nargs="+", help="Run only these scaffold IDs")
    parser.add_argument("--model", type=str, nargs="+", help="Run only these model IDs (1.7b, 4b)")
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

    # Test connection first
    print(f"Testing connection to {args.url}...")
    try:
        resp = httpx.get(f"{args.url}/health", timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Status: {health['status']}")
        print(f"  Available models: {health['available_models']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Make sure Tim has the server running: uv run server.py")
        return

    run_grid(
        base_url=args.url,
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

        print(f"Grid CRASHED: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
