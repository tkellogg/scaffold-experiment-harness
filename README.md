# Scaffold Experiment Harness

Local experiment grid for testing scaffold-centric S5 viability hypotheses.

## Requirements

- Python 3.10+
- Ollama running locally
- 32GB RAM (for qwen3:4b natural precision)

## Setup

```bash
# Install Ollama if needed
# https://ollama.ai/

# Pull models (natural precision, no quantization)
ollama pull qwen3:1.5b
ollama pull qwen3:4b

# Install dependencies
pip install httpx

# Run grid
python run_grid.py
```

## Grid Structure

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

## Usage

```bash
# Full grid (resumable)
python run_grid.py

# Specific scaffolds
python run_grid.py --scaffold 1 2 3

# Specific models
python run_grid.py --model 1.5b_think 4b_think

# Reset and start fresh
python run_grid.py --reset

# Just show summary
python run_grid.py --summary
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
