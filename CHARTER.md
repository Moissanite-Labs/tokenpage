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

## Track A: cross-language density

Earlier work found that information-dense languages such as Chinese can state some agent intentions in one or two characters. Tokenizers are built mostly around English, so the character savings did not turn into token savings.

Questions:

1. For a fixed set of agent concepts (commands, statuses, common requests), how many characters and tokens does each language take, under each tokenizer?
2. What is the cheapest form of each concept across languages, and does it change by tokenizer?
3. How much of a language's character density survives each tokenizer?
4. Which vocabulary changes would let that density survive: more merges for CJK text, single tokens for frequent short phrases, or something else?

Method:

- Reuse the study 01 pipeline: fixed traffic, pinned tokenizers, per-class attribution, a counterfactual vocabulary labeled as an estimate.
- Build a concept list from ABIOS commands and real agent logs. Write each concept in English, Chinese, Japanese, Korean and a few other languages, checked by a fluent speaker or a back-translation round trip.
- Report characters, bytes and tokens per concept, per language, per tokenizer.

Success criteria:

- A table of chars vs tokens per concept, language and tokenizer that anyone can rerun.
- A measured "density survival" ratio (token savings divided by character savings) for each tokenizer.
- A specific vocabulary change, with an estimated saving, that keeps most of the character savings.

## Track B: a symbolic agent language and its translator

Hypothesis: agents could exchange a dense symbolic language in which one symbol, possibly a new character, stands for one common thought or idea. A translator in the harness converts the user's input into that language and the model's output back into human language. This is a hypothesis to test, not a result.

Questions:

1. How many tokens does a learned symbol set save on real agent traffic, compared with ABIOS and with the cheapest language from Track A?
2. Can a small model learn to read and write the symbols with brief fine-tuning, and at what accuracy cost?
3. How accurate is the translation step, and how do its errors compound over a multi-turn session?
4. Can the translator run cheaply: first as a small hosted model that structures the user's input, later as a tiny post-trained model that runs locally on a phone or in a browser?

Method, cheapest first:

1. Measurement only: build a codebook from frequent ideas in agent logs and count tokens with and without it under stock tokenizers.
2. Add the codebook as new tokens to a small open model, fine-tune briefly, and compare task accuracy and tokens against the stock model.
3. Use a cheap hosted model as the translator. Measure round-trip accuracy (human text to symbols and back) on a fixed test set.
4. Distill a tiny translator model and measure its accuracy, size and latency on a phone-class or in-browser runtime.

Success criteria:

- A codebook that cuts tokens on agent traffic beyond ABIOS, with task accuracy within the stock model's margin.
- A translator with measured round-trip accuracy and a stated error rate per session length.
- Every symbolic message has a readable form that a human can audit.

Risks:

- Oversight. Reasoning or messages that humans cannot read are hard to audit. Every channel must keep a readable form, logged next to the symbols.
- Translator error. Mistakes in either direction compound across turns and agents. Accuracy must be measured per turn and per session, not only per message.
- New tokens need training data and fine-tuning. Savings that vanish after accuracy recovery count as a negative result.

## Training and cognition: see scry

Training and cognition work lives in [scry](https://github.com/Moissanite-Labs/scry). That includes training a model without English, or without any single human language, so it reasons natively in a symbolic language, and post-training the small translator model from Track B. The idea is that reasoning outside one language's structures could change or improve abstraction. Evidence for linguistic relativity in humans is mixed, so this would have to be measured, for example with reasoning benchmarks on symbolic-trained and English-trained small models at equal compute. This repo covers only the vocabulary and translation side.

## Out of scope

- Training a frontier model.
- Training models without human language, and reasoning benchmarks for them (see scry).
- Changing ABIOS itself. Findings that affect the protocol go to the abios repo as issues.

## Working rules

- Open-weight models and public tokenizers only, unless a provider grants access.
- Paid API spend is approved per experiment.
- Negative results get published too.
