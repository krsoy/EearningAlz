#!/bin/bash
#SBATCH --nodelist=ailab-l4-02
#SBATCH --job-name=rag_chroma
#SBATCH --output=logs/rag_chroma_%A.out
#SBATCH --error=logs/rag_chroma_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

echo "hello world"