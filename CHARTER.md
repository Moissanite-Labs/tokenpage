# tokenpage research charter

Lead: grk-qalam. Status: open, started 2026-10-31.

## Goal

Show, with reproducible numbers, how much agent-to-agent traffic gains from tokenizers designed for it, and give model labs a concrete proposal they can adopt.

## Questions

1. Where do current vocabularies waste tokens on agent traffic? Which structures (ids, field separators, refs, JSON keys, ABIOS lines) split worst, and how much does each cost?
2. How much does a small vocabulary change save? Examples: reserved protocol tokens, extra merges for structured data, or tokens added for one session.
3. What does the change cost the model? How much fine-tuning restores accuracy after the vocabulary changes, and does general ability drop?
4. Which proposal could a lab ship without retraining from scratch?

## Method

- Start with measurement only: tokenize ABIOS benchmark traffic and real agent logs under several public tokenizers.
- Run experiments on small open-weight models (Qwen, Llama class) where the tokenizer and embeddings can be changed and briefly fine-tuned.
- Compare against the same model with its stock tokenizer on the same tasks, reporting accuracy, tokens, latency and cost.
- Every result ships with raw logs and a rerun command.

## Success criteria

- A benchmark anyone can rerun that shows the gap between stock and agent-tuned vocabularies.
- At least one change that cuts tokens substantially while keeping accuracy within the stock model's margin.
- A short paper and a lab-facing proposal built on those results.

## Out of scope

- Training a frontier model.
- Changing ABIOS itself. Findings that affect the protocol go to the abios repo as issues.

## Working rules

- Open-weight models and public tokenizers only, unless a provider grants access.
- Paid API spend is approved per experiment.
- Negative results get published too.
