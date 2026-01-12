# Scaffold Experiment Harness

Local experiment grid for testing scaffold-centric S5 viability hypotheses.

## Quick Start (Mac M2)

```bash
# Install (using uv)
uv sync

# Run server (in one terminal)
uv run python server.py --port 8000

# Expose via ngrok (in another terminal)
ngrok http 8000
# → Copy the https://xxx.ngrok.io URL

# Then give the URL to Strix to hit from the bot server
```

## Server

`server.py` — Minimal MLX inference server with OpenAI-compatible endpoint.

```bash
# Start server (loads model on first request)
uv run python server.py --port 8000

# Or preload a model
uv run python server.py --preload qwen3-1.7b --port 8000
```

Uses MLX for native Apple Silicon inference. No quantization, natural precision.

## Grid Runner

`run_grid.py` — Boredom experiment grid (alternative to server approach).

**9 scaffolds × 4 model conditions × 3 runs = 108 cells**

### Scaffolds
1. baseline (no scaffolding)
2. values_only
3. values_boundaries
4. values_relationships
5. full_strix (maximum scaffolding)
6. minimal_strix ("You are Strix")
7. anti_identity (destabilizing)
8. role_technical (code reviewer archetype)
9. role_creative (creative collaborator)

### Models
- qwen3:1.5b with thinking
- qwen3:1.5b without thinking
- qwen3:4b with thinking
- qwen3:4b without thinking

## Usage (Grid)

```bash
# Full grid (resumable)
uv run python run_grid.py

# Specific scaffolds
uv run python run_grid.py --scaffold 1 2 3

# Specific models
uv run python run_grid.py --model 1.5b_think 4b_think

# Reset and start fresh
uv run python run_grid.py --reset

# Just show summary
uv run python run_grid.py --summary
```

## Output

Results saved to `results/`:
- `grid_state.json` - Grid state (resumable)
- `{scaffold}_{model}_run{n}.json` - Individual experiment logs

## Key Hypotheses

1. **Values alone may rescue collapse-prone models** — If values_only shows lower collapse than baseline, scaffolding burden is minimal
2. **Thinking architecture matters** — 1.5b_think vs 1.5b_nothink tests whether thinking tokens provide stability
3. **Anti-identity can break stable models** — If stable models collapse under anti_identity, scaffolding can destabilize
4. **Role-based specs work for productization** — If role_technical/creative show stability, factory templates are viable
