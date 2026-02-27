#!/bin/bash
uv sync
uv run python -m ipykernel install --user --name=tp-embeddings --display-name "TP Embeddings (uv)"
echo "Env ready"