#!/usr/bin/env bash
set -euo pipefail

echo "Starting GPU test loop..."

for i in {1..5}; do
    echo ""
    echo "=== Iteration $i ==="

    # Call Llama-3-8B API with a simple prompt
    echo "Calling Llama-3-8B..."
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama3",
        "messages": [
          {"role": "user", "content": "Hello, summarize the key points of quantum computing in 3 sentences."}
        ]
      }' | jq '.choices[].message.content'

    echo ""

    # Run your profile binary
    ./profile diagnose
    
    echo "Nvidia-smi..."
    nvidia-smi

    echo "Sleeping 2 seconds before next iteration..."
    sleep 2
done

echo "GPU test loop complete."
