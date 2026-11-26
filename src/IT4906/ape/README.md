# APE (Automatic Prompt Engineer)

## Algorithm

1. Load the 10 distinct prompts from `prompts_auto.txt` as the initial population P0.
2. Generate Findings with MedGemma and compute fitness for each prompt on the same 50 validation samples: `0.7 × BERTScore F1 + 0.3 × ROUGE-L F1`.
3. Select the top 10 prompts as the current elite population.
4. Ask Qwen3.5-9B through the Evolution API to paraphrase each elite prompt into one offspring.
5. Generate Findings and score each offspring on those same validation samples.
6. Combine the old elites and offspring, then retain the top 10 by fitness. Stable sorting gives existing elites priority on ties.
7. Repeat for the configured generation budget. The notebook default is 2 generations, which evaluates 10 initial prompts and 20 offspring (30 fitness evaluations total).
8. Freeze the highest-fitness validation prompt and evaluate only that prompt on the complete 540-row test split. Report test BERTScore, ROUGE-L, Corpus BLEU, and Fitness.

Do not use test data during paraphrasing or selection. Each extra generation costs 10 more prompt evaluations × 50 validation samples.

## Kaggle setup

Use [the APE notebook on Kaggle](https://www.kaggle.com/code/drakan02/it4906-ape-50-val-and-full-test). Add private runtime dataset `drakan02/it4906-ape-runtime`. Choose T4 ×2, enable Internet, and add these Kaggle Secrets for the notebook:

- `HF_TOKEN` — Hugging Face read token with access to MedGemma.
- `EVOLUTION_LLM_BASE_URL` and `EVOLUTION_LLM_API_KEY` — Evolution endpoint used by the original pipeline.

The notebook outputs `candidate_scores.json`, `population.json`, `best_prompt.json`, validation checkpoints, test metrics/predictions, `summary.csv`, and a downloadable ZIP.
