# Line-pinned mutation campaign for the live-trading safety paths.
# This file is not included by the repo Makefile.
# Run: make -f deploy/mutation.mk mutation-campaign

.PHONY: mutation-campaign

mutation-campaign:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
	VECLIB_MAXIMUM_THREADS=1 KMP_INIT_AT_FORK=FALSE \
	PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false \
	MLFLOW_DISABLE_AGENT_HINT=1 \
	uv run python scripts/mutation_campaign.py --min-score 0.90
