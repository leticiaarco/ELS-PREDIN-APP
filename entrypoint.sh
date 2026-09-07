#!/bin/bash

# Start the Ollama server in the background
/bin/ollama serve &

# Wait until Ollama API becomes responsive using native ollama list command
echo "Waiting for Ollama to initialize..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done

# Pull the required LLM model automatically
echo "Pulling LLM model: qwen2.5:3b (this will download on first run only)..."
ollama pull qwen2.5:3b

echo "Ollama setup complete! System ready."

# Keep the container running in the foreground
wait -n