import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import sys
import os
import argparse


# -----------------------------
# Experiment configuration
# -----------------------------

@dataclass
class ExperimentConfig:
    retriever: str                  # dense | bm25 | hybrid
    reranker: bool = False
    chunking: Dict[str, Optional[int | str]] | None = None


# -----------------------------
# Chunking experiment grid
# -----------------------------

CHUNKING_GRID = [
    {"strategy": "fixed", "max_chars": 500,  "overlap": 100},
    {"strategy": "fixed", "max_chars": 1000, "overlap": 200},
    {"strategy": "fixed", "max_chars": 2000, "overlap": 400},
    {"strategy": "sentence"},
    {"strategy": "section"},
]


def build_chunking_experiments() -> List[ExperimentConfig]:
    """
    Run experiments where ONLY chunking varies.
    All other components are fixed.
    """
    return [
        ExperimentConfig(
            retriever="dense",
            reranker=False,
            chunking=chunk_cfg,
        )
        for chunk_cfg in CHUNKING_GRID
    ]


# -----------------------------
# Runner utilities
# -----------------------------

def run_command(cmd: List[str], env: dict):
    full_cmd = [sys.executable] + cmd[1:]
    print(" ".join(full_cmd))
    subprocess.run(full_cmd, check=True, env=env)


def experiment_tag(exp: ExperimentConfig) -> str:
    chunk = exp.chunking or {}
    strat = chunk.get("strategy", "default")

    if strat == "fixed":
        chunk_tag = f"fixed_{chunk['max_chars']}_{chunk['overlap']}"
    else:
        chunk_tag = strat

    return f"{chunk_tag}_{exp.retriever}_{'rerank' if exp.reranker else 'norank'}"


# -----------------------------
# Main orchestration
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner")

    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Run chunking ablation experiments",
    )

    args = parser.parse_args()

    if not args.chunking:
        raise ValueError(
            "No experiment type specified. "
            "Use --chunking to run chunking experiments."
        )

    experiments = build_chunking_experiments()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("evaluation/results") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {out_dir}\n")

    for exp in experiments:
        tag = experiment_tag(exp)
        print(f"\n=== Running experiment: {tag} ===\n")

        # Inject chunking config via environment
        env = os.environ.copy()
        env["CHUNK_STRATEGY"] = exp.chunking["strategy"]
        env["CHUNK_MAX_CHARS"] = str(exp.chunking.get("max_chars", ""))
        env["CHUNK_OVERLAP"] = str(exp.chunking.get("overlap", ""))

        rag_cmd = [
            "python", "-m", "evaluation.run_rag_eval",
            "--retriever", exp.retriever,
        ]

        if exp.reranker:
            rag_cmd.append("--reranker")

        run_command(rag_cmd, env)


if __name__ == "__main__":
    main()
