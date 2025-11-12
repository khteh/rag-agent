#!/bin/bash
pushd data
kubectl cp Healthcare neo4j-0:import
popd
uv run python -m src.utils.Neo4JImports