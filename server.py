#!/usr/bin/env python3
"""Multi-model MLX inference server for Mac M2.

Serves both Qwen3-1.7B and Qwen3-4B on the same port.
Models are loaded on-demand and hot-swapped based on request.

Usage:
    pip install mlx-lm flask
    python server.py --port 8000

Then expose via ngrok:
    ngrok http 8000

Request with model param to select:
    {"model": "qwen3-1.7b", "messages": [...]}
    {"model": "qwen3-4b", "messages": [...]}
"""
import argparse
from flask import Flask, request, jsonify

app = Flask(__name__)

# Model registry - native bf16, no quantization
MODEL_REGISTRY = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-MLX-bf16",
    "qwen3-4b": "Qwen/Qwen3-4B-MLX-bf16",
    # Aliases for flexibility
    "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B-MLX-bf16",
    "Qwen/Qwen3-4B": "Qwen/Qwen3-4B-MLX-bf16",
    "Qwen/Qwen3-1.7B-MLX-bf16": "Qwen/Qwen3-1.7B-MLX-bf16",
    "Qwen/Qwen3-4B-MLX-bf16": "Qwen/Qwen3-4B-MLX-bf16",
}

# Current loaded model state
CURRENT_MODEL = None
CURRENT_TOKENIZER = None
CURRENT_MODEL_ID = None


def load_model(model_key: str):
    """Load model using MLX (Apple Silicon optimized, native bf16)."""
    global CURRENT_MODEL, CURRENT_TOKENIZER, CURRENT_MODEL_ID
    from mlx_lm import load

    # Resolve model key to HuggingFace ID
    hf_model_id = MODEL_REGISTRY.get(model_key, model_key)

    # Skip if already loaded
    if CURRENT_MODEL_ID == hf_model_id:
        return

    # Unload previous model to free memory
    if CURRENT_MODEL is not None:
        print(f"Unloading {CURRENT_MODEL_ID}...")
        CURRENT_MODEL = None
        CURRENT_TOKENIZER = None
        # Force garbage collection
        import gc
        gc.collect()

    print(f"Loading {hf_model_id} (native bf16, no quantization)...")
    CURRENT_MODEL, CURRENT_TOKENIZER = load(hf_model_id)
    CURRENT_MODEL_ID = hf_model_id
    print(f"Model loaded: {hf_model_id}")


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint with model hot-swap."""
    from mlx_lm import generate

    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 500)
    requested_model = data.get("model", "qwen3-1.7b")

    # Hot-swap model if needed
    load_model(requested_model)

    # Build prompt from messages using chat template
    if CURRENT_TOKENIZER.chat_template is not None:
        prompt = CURRENT_TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback to manual formatting
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

    # Generate
    response = generate(
        CURRENT_MODEL,
        CURRENT_TOKENIZER,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    return jsonify({
        "id": "local-chat",
        "object": "chat.completion",
        "model": CURRENT_MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "current_model": CURRENT_MODEL_ID,
        "available_models": list(set(MODEL_REGISTRY.values()))
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible models list endpoint."""
    models = list(set(MODEL_REGISTRY.values()))
    return jsonify({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "local"}
            for m in models
        ]
    })


def main():
    parser = argparse.ArgumentParser(description="Multi-model MLX inference server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--preload", help="Preload a model at startup (optional)")
    args = parser.parse_args()

    if args.preload:
        load_model(args.preload)

    print(f"\nServer running on http://localhost:{args.port}")
    print(f"Available models: {list(set(MODEL_REGISTRY.values()))}")
    print(f"\nExpose via: ngrok http {args.port}")
    print("\nModels load on first request. Hot-swap via 'model' param in request.")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
