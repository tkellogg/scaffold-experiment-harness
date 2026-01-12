#!/usr/bin/env python3
"""Minimal inference server for Mac M2.

Usage:
    pip install mlx-lm flask
    python server.py --model Qwen/Qwen3-1.7B --port 8000
    python server.py --model Qwen/Qwen3-4B --port 8000

Then expose via ngrok:
    ngrok http 8000
"""
import argparse
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global model/tokenizer - loaded once at startup
MODEL = None
TOKENIZER = None
MODEL_NAME = None


def load_model(model_name: str):
    """Load model using MLX (Apple Silicon optimized)."""
    global MODEL, TOKENIZER, MODEL_NAME
    from mlx_lm import load

    print(f"Loading {model_name}...")
    MODEL, TOKENIZER = load(model_name)
    MODEL_NAME = model_name
    print(f"Model loaded: {model_name}")


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    from mlx_lm import generate

    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 500)

    # Build prompt from messages
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
        MODEL,
        TOKENIZER,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    return jsonify({
        "id": "local-chat",
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,  # MLX doesn't expose this easily
            "completion_tokens": 0,
            "total_tokens": 0
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


def main():
    parser = argparse.ArgumentParser(description="Minimal MLX inference server")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HuggingFace model ID")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    load_model(args.model)
    print(f"\nServer running on http://localhost:{args.port}")
    print(f"Expose via: ngrok http {args.port}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
