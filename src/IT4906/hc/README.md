# Hill Climbing (Iterative Prompt Refinement)

## Method

1. Select one seed prompt from the 10 distinct lines in `prompts_auto.txt` using seed 5.
2. Use MedGemma to generate Findings for the same 50 validation samples and score with `0.7 × BERTScore F1 + 0.3 × ROUGE-L F1`.
3. Ask Qwen3.5-9B, through the Evolution API used by `it4906-pipeline.ipynb`, to propose a focused revision of the current prompt.
4. Evaluate that proposal on the same validation samples. Replace the current prompt only if the proposal's fitness is strictly higher; otherwise retain the current prompt.
5. Repeat for 9 refinement attempts. This gives 10 fitness evaluations total including the initial prompt, matching the Random Search evaluation budget.
6. Freeze the final prompt and evaluate it once on all 540 test samples. Use test BERTScore, ROUGE-L, Corpus BLEU, and Fitness for the report table.

The held-out test set is never used to propose or select prompts. Every proposal and acceptance decision is recorded in `hill_climbing_state.json` and `candidate_scores.json`.

## Kaggle setup

Use [the Hill Climbing notebook on Kaggle](https://www.kaggle.com/code/drakan02/it4906-hill-climbing-50-val-and-full-test). Add the private runtime dataset, select T4 ×2, and enable Internet. Add these Kaggle Secrets and grant this notebook access:

- `HF_TOKEN` — read access to MedGemma, after accepting Hugging Face gated-model terms.
- `EVOLUTION_LLM_BASE_URL` and `EVOLUTION_LLM_API_KEY` — Evolution API credentials used by the original pipeline.

The notebook writes `summary.csv`, predictions, validation checkpoints, logs, and a downloadable ZIP under Kaggle Output. The test metric row is the value to copy into the report.
