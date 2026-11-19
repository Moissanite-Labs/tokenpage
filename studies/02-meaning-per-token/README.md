# Study 02: meaning recovered per token

This study asks how much of a message's meaning a model can get back per token, for five ways of writing the same message. It uses Qwen2.5 0.5B and 1.5B on CPU and calls no paid API. Every number here comes from `results/`.

## Question

Token counts alone say which format is cheap. They do not say whether the model still gets the meaning. Which format gives the most recovered meaning per token, and can a model read a dense symbolic dialect it was never trained on?

## Design

**Messages.** 600 synthetic `task.create` messages (`messages.py`). Each has 5 independent, uniformly drawn fields, 11.49 bits in total: title (12 options), priority (4), scope (6), acceptance criterion (5) and depends-on-task (2).

**Formats.** The same 600 messages in 5 renderings:

- `english`: 3 phrasing templates.
- `json`: compact JSON with short refs `p` and `t`.
- `abios_refs`: the ABIOS/3 reference encoder at abios commit `6da3a64`, refs family, fresh session per message.
- `chinese`: 2 templates. The Chinese was written by a non-native speaker.
- `glyph_g1`: a symbolic dialect written for this study, e.g. `+⟨csv-export⟩‼@db✓H←t`. English slugs for titles, one glyph or letter per enumerated value.

**Measures.**

1. **State probe.** Qwen2.5-0.5B-Instruct hidden states, mean-pooled over the message tokens. Standardize, PCA(128) fit on the train fold, L2 logistic regression (C = 0.05), 5-fold CV. Bits recovered per field = log2(K) minus held-out cross-entropy in bits, floored at 0, summed over fields. Layer 16 of 24 was chosen in advance. Layers 4, 8, 12, 20 and 24 are in `results/`.
2. **Bag-of-tokens baseline.** The same probe on token unigram and bigram counts.
3. **Transfer.** A probe trained on English states, tested on each other format. "Raw" has no adaptation. "Centered" subtracts each format's unlabeled mean first.
4. **Reading test.** Qwen2.5-1.5B-Instruct answers one multiple-choice question per field, scored by option log-likelihood. 24 messages per condition. 7 conditions: the 5 formats with no legend, plus ABIOS and G1 with a one-paragraph legend.
5. **Token counts** under Qwen2.5, o200k_base and cl100k_base.

The research plan proposed a ridge-regression probe scored by fraction of variance explained. This study uses the classifier probe above instead, because the fields are categorical and bits are comparable across fields.

## Results

### Token counts (`results/results_token_counts.txt`)

| format | qwen2.5 tok | o200k tok | cl100k tok |
|---|---|---|---|
| english | 28.4 | 28.1 | 28.4 |
| json | 34.9 | 37.2 | 34.9 |
| abios_refs | 36.6 | 36.5 | 36.6 |
| chinese | 29.8 | 34.5 | 46.7 |
| glyph_g1 | 11.8 | 15.7 | 14.0 |

ABIOS refs messages average 21.4 Qwen tokens without a dependency and 54.5 with one (`results/results_tokens_by_dependency.txt`). The dependency is a full 32-character hex id, because each message is encoded in a fresh session.

### State probe, layer 16 (`results/results_probe_qwen0.5b_L16.txt`)

| format | qwen tokens/msg | bits recovered | bits/token | bag-of-tokens bits | EN->fmt transfer bits (raw) | EN->fmt transfer bits (centered) |
|---|---|---|---|---|---|---|
| english | 28.4 | 11.21 | 0.395 | 11.22 | 11.21 | 11.21 |
| json | 34.9 | 11.37 | 0.326 | 11.26 | 3.08 | 8.33 |
| abios_refs | 36.6 | 10.95 | 0.299 | 11.27 | 2.32 | 5.90 |
| chinese | 29.8 | 11.33 | 0.380 | 11.24 | 4.59 | 7.19 |
| glyph_g1 | 11.8 | 11.32 | 0.962 | 11.17 | 1.34 | 4.23 |

### Reading test (`results/results_qa_qwen1.5b.md`)

| format | legend | title acc | priority acc | scope acc | criteria acc | dependency acc | mean acc | bits recovered (sum over fields, H - CE of option probs) |
|---|---|---|---|---|---|---|---|---|
| english | False | 1.00 | 1.00 | 0.92 | 1.00 | 0.75 | 0.933 | 10.69 |
| json | False | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 11.48 |
| abios_refs | False | 1.00 | 1.00 | 0.96 | 1.00 | 0.50 | 0.892 | 10.19 |
| chinese | False | 1.00 | 1.00 | 1.00 | 1.00 | 0.79 | 0.958 | 11.09 |
| glyph_g1 | False | 0.88 | 0.50 | 0.50 | 0.21 | 0.71 | 0.558 | 2.58 |
| abios_refs | True | 1.00 | 1.00 | 0.88 | 1.00 | 0.71 | 0.917 | 10.10 |
| glyph_g1 | True | 0.25 | 0.58 | 0.54 | 0.42 | 0.92 | 0.542 | 0.62 |

Chance: title 0.08, priority 0.25, scope 0.17, criteria 0.20, dependency 0.50. n = 24 per cell.

Reading bits per Qwen token, from the two tables above: english 0.376, json 0.329, chinese 0.372, abios_refs 0.278 (0.276 with legend), glyph_g1 0.219 (0.053 with legend, legend tokens not counted).

## What it shows

- Every format is near-fully decodable from the hidden states: 10.95 to 11.37 of 11.49 bits. On the probe, G1 gets 0.96 bits per Qwen token against 0.40 for English.
- The probe only shows the information is there. A probe trained on English recovers 4.23 of 11.49 bits from G1 after centering. The model does not put G1 meaning where it puts English meaning.
- The reading test shows the model cannot use G1. Without the legend, 1.5B recovers 2.58 of 11.49 bits. With the legend it recovers 0.62. Title accuracy falls from 0.88 to 0.25, and 19 of 24 title answers with the legend are one of the first two options.
- On reading, English, JSON and Chinese lead on bits per token. JSON is read perfectly.
- ABIOS refs come to 0.30 bits per token on the probe for these short flat messages. The full hex id adds about 33 tokens to messages with a dependency (21.4 vs 54.5). The dependency field is also the one 1.5B misreads (0.50 without the legend; with it, every answer is "no").
- Chinese gives no density gain. It costs more tokens than English on o200k (34.5 vs 28.1) and on cl100k (46.7 vs 28.4).

## What it does not show

- The probe saturates. All formats reach about 11 bits, and so does the bag-of-tokens baseline (11.17 to 11.27). So probe bits per token mostly tracks token count. The reading test is where the formats differ.
- The messages are short, flat and synthetic, with uniform fields. Real agent traffic has longer messages, shared context and skewed values.
- ABIOS is measured one message per fresh session. In a long session, refs and aliases make ids cheaper.
- Only small Qwen models were tested, with 24 messages per reading condition. A larger model, or one trained on G1, may read it.
- The Chinese text was written by a non-native speaker.
- G1 was designed by the same author who ran the study.

## Rerun

From this folder. `abios_refs` needs an abios checkout at commit `6da3a64`; point `ABIOS_DIR` at it. `torch==2.15.0` is the CPU build.

```
pip install -r requirements.txt
export ABIOS_DIR=/path/to/abios
python extract.py Qwen/Qwen2.5-0.5B-Instruct 600 states_qwen0.5b.npz > results/log_extract_qwen0.5b.txt 2>&1
for L in 4 8 12 16 20 24; do OMP_NUM_THREADS=1 python -u probe.py states_qwen0.5b.npz $L > results/results_probe_qwen0.5b_L$L.txt; done
python token_counts.py > results/results_token_counts.txt
python tokens_by_dependency.py > results/results_tokens_by_dependency.txt
python -u qa.py Qwen/Qwen2.5-1.5B-Instruct 24 results/raw_qa_qwen1.5b.jsonl >> results/log_qa_qwen1.5b.txt 2>&1
python qa_summary.py results/raw_qa_qwen1.5b.jsonl 24 > results/results_qa_qwen1.5b.md
```

`extract.py` writes `states_qwen0.5b.npz` (about 250 MB), which is not committed. It takes about 2.5 minutes, each probe layer about 1 minute, and the reading test about 35 minutes on CPU. `qa.py` resumes from the rows already in its output file, so delete `results/raw_qa_qwen1.5b.jsonl` first for a fresh run. The reading test also writes extra English rows past message 24; `qa_summary.py` ignores them.

Versions used: numpy 2.5.4, scikit-learn 1.9.1, torch 2.15.0+cpu, transformers 5.22.1, tiktoken 0.14.0.
