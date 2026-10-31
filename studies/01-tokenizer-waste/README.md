# Study 01: where tokens go in ABIOS traffic

This study measures how seven public tokenizers split the ABIOS benchmark traffic. It does not train anything and calls no paid API. Every number here comes from `results/`, which `measure.py` regenerates.

## Rerun

```
pip install -r requirements.txt && python measure.py
```

That reads the committed `data/streams.json` and rewrites `results/`. Runtime is about 5 s after the tokenizer downloads. To rebuild `data/streams.json` from abios, you need an abios tree at commit `6da3a64`:

```
python extract_streams.py --abios /path/to/abios --commit 6da3a64
```

## Method

**Traffic.** The three bench tasks from abios `scripts/protean-abios/bench.py`: `lifecycle_11`, `lifecycle_40` and `lifecycle_200`. `extract_streams.py` imports the abios code and writes three encodings of each task:

- **ABIOS refs.** One line per message. Setup is the POST ack plus the minimal setup text. This is the same accounting as the bench: `tok(post_ack + minimal_text) + tok("\n".join(lines))`.
- **Compact JSON.** `json.dumps({"action": a, **params}, sort_keys=True, separators=(",", ":"))`, one line per message, no setup.
- **English lossless.** abios `render_english_lossless`, one line per message, no setup.

**Source.** The abios checkout on this machine was at `31d90a3`, not `6da3a64`. I exported `6da3a64` with `git archive` into a separate directory and did not touch the checkout. `data/streams.json` records the commit (`abios_commit_seen: 6da3a64`).

**Sanity check.** `measure.py` asserts the published o200k totals before it writes anything. All six match.

| task | ABIOS (expected / measured) | compact JSON (expected / measured) |
| --- | ---: | ---: |
| lifecycle_11 | 761 / 761 | 738 / 738 |
| lifecycle_40 | 1983 / 1983 | 5570 / 5570 |
| lifecycle_200 | 7791 / 7791 | 28498 / 28498 |

**Tokenizers.** No special tokens are added. HF tokenizers are pinned to a commit. `measure.py` stops if a downloaded `tokenizer.json` does not match the pinned sha256, and records the full hashes in `results/results.json`. For tiktoken, the hash column is the BPE file hash that tiktoken itself checks.

| name | source | revision | tokenizer.json sha256 (prefix) | vocab |
| --- | --- | --- | --- | ---: |
| o200k_base | tiktoken 0.9.0 | built in; BPE file sha256 checked by tiktoken | 446a9538cb6c348e | 200019 |
| cl100k_base | tiktoken 0.9.0 | built in; BPE file sha256 checked by tiktoken | 223921b76ee99bde | 100277 |
| llama3 | NousResearch/Meta-Llama-3.1-8B-Instruct | d10aef7999a2b5ba950ab3974312feeedbfe0b77 | 79e3e522635f3171 | 128256 |
| qwen3 | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | aeb13307a71acd8f | 151669 |
| mistral_nemo | mistralai/Mistral-Nemo-Instruct-2407 | 04d8a90549d23fc6bd7f642064003592df51e9b3 | e11c71726323d33d | 131072 |
| gemma3 | unsloth/gemma-3-1b-it | 5b11413a10db4e486ef16a20101fd028f8f2499c | 4667f2089529e8e7 | 262145 |
| deepseek_v3 | deepseek-ai/DeepSeek-V3 | e815299b0bcbac849fa540c768ef21845365c9eb | 621ac2e32d0dba65 | 128815 |

Llama 3 uses an ungated mirror of the official tokenizer. `google/gemma-3-1b-it` is gated, so Gemma uses the unsloth mirror. Neither mirror was checked byte for byte against the gated original.

**Attribution.** Each character of the stream gets a structural class. Each token's weight is split evenly over the characters it covers, using the tokenizer's own offsets. A class's token count is the sum of its characters' weights. So a token like ` test` puts half its weight on the space (field separator) and half on `test`. Every character was covered for every tokenizer (`offset_coverage` in `results.json`).

**Worst-splitting units.** A unit is a recurring piece of a line: `E`+command, one ` field`, or a run of quotes, escapes and JSON brackets. Its tokens come from the fractional attribution above. Its "ideal" cost is one token for a ref, number, enum or command; one token per word for prose; and one token per 4 hex chars (16 bits) for ids. Excess = occurrences × (mean tokens across tokenizers − ideal).

**Counterfactual.** This is an estimate of a vocabulary change, not a trained tokenizer. Messages are cut into pieces. Some pieces count as one token and the rest are tokenized on their own with the stock tokenizer. Setup stays stock. The variants:

- **A.** The format tag, each lexicon word, each field separator, each newline and each `#n` (n < 1000) count as one token. This is the literal brief.
- **B.** The format tag, each lexicon word and each ` #n` (with its leading space) count as one token. Other ` field` pieces keep their space and use stock tokens.
- **C.** B, plus each run of structural quotes, escapes and JSON brackets counts as one token.

Cutting text into pieces itself changes the count, because tokens can no longer span a cut. So each variant also reports a "segmented stock" count: the same pieces, all with stock tokens. "Saved vs segmented" isolates the vocabulary effect. "Saved vs stock" is the number you would see.

## Results

### Totals

Tokens for the whole task. ABIOS includes setup.

| tokenizer | 11: ABIOS / JSON / English | 40: ABIOS / JSON / English | 200: ABIOS / JSON / English | ABIOS / JSON at 200 |
| --- | ---: | ---: | ---: | ---: |
| o200k_base | 761 / 738 / 686 | 1983 / 5570 / 5242 | 7791 / 28498 / 26754 | 0.273 |
| cl100k_base | 752 / 737 / 688 | 1987 / 5632 / 5284 | 7855 / 28700 / 26960 | 0.274 |
| llama3 | 752 / 737 / 688 | 1987 / 5632 / 5284 | 7855 / 28700 / 26960 | 0.274 |
| qwen3 | 758 / 738 / 689 | 2021 / 5659 / 5311 | 7997 / 28835 / 27095 | 0.277 |
| mistral_nemo | 870 / 848 / 780 | 2489 / 6795 / 6317 | 10085 / 34515 / 32125 | 0.292 |
| gemma3 | 834 / 808 / 765 | 2114 / 6240 / 5976 | 8194 / 31744 / 30424 | 0.258 |
| deepseek_v3 | 831 / 802 / 737 | 2096 / 6218 / 5821 | 8148 / 31630 / 29645 | 0.258 |

**Crossover.** On `lifecycle_11`, ABIOS never beats JSON or English for any tokenizer. On `lifecycle_40` and `lifecycle_200`, ABIOS beats both from message 12 on, for every tokenizer. The first `#n` ref appears in message 9, and the first exact repeat of a whole line is message 15. By message 12 the shorter ABIOS lines have paid back the setup. The crossover point depends on the fixture as much as on the encoding.

On `lifecycle_11`, ABIOS messages alone cost 236 tokens (o200k) against 738 for JSON. Setup costs 525 and decides the result.

### Where ABIOS tokens go

`lifecycle_200`, share of ABIOS tokens / chars per token. Full tables for every tokenizer and for `lifecycle_11` are in `results/tables.md`.

| class | chars | o200k_base | qwen3 | mistral_nemo | deepseek_v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| payload prose | 12425 | 34.8% / 4.59 | 34.8% / 4.46 | 28.0% / 4.39 | 34.8% / 4.38 |
| hex ids / digests | 6064 | 12.7% / 6.12 | 12.4% / 6.12 | 29.3% / 2.05 | 12.3% / 6.03 |
| field separator (space) | 1715 | 9.6% / 2.29 | 9.4% / 2.29 | 7.4% / 2.29 | 9.5% / 2.22 |
| quotes and backslash escapes | 990 | 7.9% / 1.62 | 7.7% / 1.62 | 6.1% / 1.61 | 7.9% / 1.54 |
| content refs #n | 548 | 5.3% / 1.33 | 5.1% / 1.33 | 4.1% / 1.33 | 5.0% / 1.33 |
| numbers (` 3.14`) | 540 | 5.2% / 1.33 | 6.8% / 1.00 | 5.3% / 1.00 | 5.0% / 1.33 |
| enum values | 1905 | 4.2% / 5.90 | 4.0% / 5.90 | 3.4% / 5.55 | 4.2% / 5.55 |
| JSON structure inside a field | 720 | 4.0% / 2.31 | 3.9% / 2.31 | 3.1% / 2.30 | 5.0% / 1.76 |
| lexicon command word | 870 | 2.6% / 4.24 | 2.6% / 4.24 | 2.1% / 4.14 | 2.6% / 4.14 |
| format tag E | 200 | 2.5% / 1.03 | 2.4% / 1.03 | 1.9% / 1.03 | 2.4% / 1.03 |
| line separator | 199 | 2.4% / 1.08 | 2.3% / 1.08 | 1.8% / 1.08 | 2.3% / 1.08 |
| session handles p/t/a/h/f | 335 | 2.1% / 2.00 | 2.1% / 2.00 | 1.7% / 2.00 | 2.1% / 2.00 |
| setup (ack, header, lexicon, schema, ids, defaults) | 1683 | 6.8% | 6.5% | 5.7% | 6.9% |

`lifecycle_11`, o200k: setup is 69% of the ABIOS total. The ids block alone is 241 chars at 1.32 chars per token, 24.1% of all tokens (about 183). The schema block is 17.9% and the lexicon block 14.3%.

### Where JSON tokens go

`lifecycle_200`, share of JSON tokens / chars per token.

| class | chars | o200k_base | qwen3 | mistral_nemo | deepseek_v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hex ids / digests | 26000 | 48.4% / 1.89 | 47.8% / 1.89 | 53.2% / 1.42 | 50.9% / 1.61 |
| payload prose | 29155 | 22.7% / 4.52 | 23.3% / 4.35 | 20.3% / 4.16 | 21.8% / 4.24 |
| keys (with quotes and colon) | 19095 | 14.4% / 4.65 | 14.1% / 4.69 | 13.2% / 4.20 | 14.1% / 4.28 |
| punctuation and value quotes | 8260 | 9.7% / 2.99 | 9.5% / 3.02 | 8.5% / 2.82 | 9.1% / 2.86 |
| enum values | 4390 | 3.2% / 4.80 | 3.2% / 4.77 | 3.1% / 4.12 | 2.6% / 5.35 |
| numbers (`"3.14"`) | 540 | 1.4% / 1.33 | 1.9% / 1.00 | 1.6% / 1.00 | 1.3% / 1.33 |
| line separator | 199 | 0.2% / 3.00 | 0.2% / 3.00 | 0.2% / 3.00 | 0.2% / 3.00 |

### Top 20 recurring units by excess tokens (ABIOS, lifecycle_200)

Tokens per occurrence. Fractional values come from tokens that span a unit boundary. Structural runs (`"[\"`, `\"]" "`) overlap the quoted fields they belong to.

| unit | kind | occ | ideal | o200k | cl100k | llama3 | qwen3 | nemo | gemma3 | deepseek | excess |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ` "python -m unittest discover -s tests -v"` | quoted_prose | 135 | 7 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 675 |
| ` 3.14` | number | 130 | 1 | 4 | 4 | 4 | 5 | 5 | 5 | 4 | 446 |
| ` "[\"CI not executed in transcript.\"]"` | quoted_json | 75 | 5 | 10 | 10 | 10 | 10 | 10 | 11 | 11 | 396 |
| `\"]" "` | structural_run | 135 | 1 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 338 |
| ` "[\"Synthetic bench filler.\"]"` | quoted_json | 55 | 3 | 8 | 9 | 9 | 9 | 9 | 9 | 10 | 330 |
| ` "[\"` | structural_run | 140 | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 180 |
| ` #1` | content_ref | 144 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 144 |
| `Etest` | tag+cmd | 135 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 135 |
| ` "[\"Round-trip lossless\",\"Validator exit 0\"]"` | quoted_json | 5 | 6 | 13 | 14 | 14 | 14 | 15 | 14 | 17 | 42 |
| ` "[\"Reported by agent; not executed by forge.\"]"` | quoted_json | 5 | 7 | 13 | 14 | 14 | 14 | 14 | 15 | 15 | 36 |
| ` ["claim","work","artifact","evidence","submit"]` | bare_json | 5 | 5 | 12 | 12 | 12 | 12 | 12 | 11 | 11 | 34 |
| ` ["synbios/sym2p"]` | bare_json | 5 | 2 | 8 | 8 | 8 | 8 | 9 | 8 | 9 | 31 |
| ` "Encode forge commands as SYM-2P v1 packets."` | quoted_prose | 5 | 8 | 14 | 14 | 14 | 14 | 14 | 13 | 15 | 30 |
| ` #2` | content_ref | 30 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 30 |
| ` "https://example.invalid/sym3"` | quoted_prose | 5 | 4 | 8.50 | 8.50 | 8.50 | 8.50 | 9.50 | 10 | 9.50 | 25 |
| ` #3` | content_ref | 25 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 25 |
| ` #4` | content_ref | 25 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 25 |
| ` #5` | content_ref | 25 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 25 |
| ` #6` | content_ref | 25 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 25 |
| `" ` | structural_run | 160 | 1 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 24 |

The top JSON units are the fixture ids: `"b2b2…"` (180 times) and `"e5e5…"` (140 times) cost 32.7 tokens each in every tokenizer, against an ideal of 8. See `results/tables.md`.

### Probes

Tokens for one string tokenized alone.

| text | o200k | cl100k | llama3 | qwen3 | nemo | gemma3 | deepseek |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ` #1` | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| ` #12` | 2 | 2 | 2 | 3 | 3 | 3 | 2 |
| ` #123` / ` #999` | 2 | 2 | 2 | 4 | 4 | 4 | 2 |
| fixture id, 16 × `a1` | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| random 32 hex | 17 | 17 | 17 | 27 | 28 | 27 | 18 |
| fixture revision, 40 × `a` | 7 | 7 | 7 | 7 | 20 | 7 | 7 |
| random 40 hex | 19 | 19 | 19 | 34 | 34 | 34 | 19 |
| fixture digest, 64 × `b` | 17 | 17 | 17 | 17 | 33 | 32 | 32 |
| random 64 hex | 36 | 36 | 36 | 54 | 56 | 54 | 37 |
| ` "[\"` (JSON list opener in a quoted field) | 2 | 2 | 2 | 2 | 2 | 3 | 3 |
| `\"]"` (closer) | 2 | 2 | 2 | 2 | 3 | 2 | 3 |

### Counterfactual vocabulary (estimate, not a trained result)

Totals include stock setup. "Saved vs stock" is against the real ABIOS total. Only ABIOS gets a counterfactual. JSON gets no matching change (for example, single-token keys), so these numbers do not say how the ABIOS/JSON gap would move if both encodings got new tokens.

| tokenizer | 200: stock | A saved | B saved | C saved | B saved vs segmented | C saved vs segmented | 11: B saved | 11: C saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| o200k_base | 7791 | -1020 (-13.1%) | 244 (3.1%) | 494 (6.3%) | 274 | 554 | -4 (-0.5%) | -5 (-0.7%) |
| cl100k_base | 7855 | -1020 (-13.0%) | 244 (3.1%) | 494 (6.3%) | 274 | 554 | -4 (-0.5%) | -5 (-0.7%) |
| llama3 | 7855 | -1020 (-13.0%) | 244 (3.1%) | 494 (6.3%) | 274 | 554 | -4 (-0.5%) | -5 (-0.7%) |
| qwen3 | 7997 | -1020 (-12.8%) | 244 (3.0%) | 494 (6.2%) | 274 | 554 | -4 (-0.5%) | -5 (-0.7%) |
| mistral_nemo | 10085 | -1580 (-15.7%) | 249 (2.5%) | 504 (5.0%) | 279 | 699 | -3 (-0.3%) | -3 (-0.3%) |
| gemma3 | 8194 | -1006 (-12.3%) | 274 (3.3%) | 664 (8.1%) | 274 | 694 | 1 (0.1%) | 2 (0.2%) |
| deepseek_v3 | 8148 | -1031 (-12.7%) | 249 (3.1%) | 649 (8.0%) | 279 | 844 | -3 (-0.4%) | 0 (0.0%) |

Lexicon words alone (B with only the command word as one token), lifecycle_200, saved vs segmented stock: o200k_base 0, cl100k_base 0, llama3 0, qwen3 0, mistral_nemo 5, gemma3 0, deepseek_v3 5.

Under B, the whole gain comes from refs. Each ` #n` drops from 2 tokens to 1, and `lifecycle_200` has 274 refs. Under A, isolating every separator and newline costs 1006 to 1580 extra tokens. In stock tokenization the space before a field shares a token with the next word, so it costs almost nothing. Splitting it off makes it a full token.

The random-hex rerun below changes these baselines but not the counted pieces. The full table, with `lifecycle_40`, is in `results/tables.md`.

### Random-hex sensitivity

The fixtures use low-entropy hex: revisions are 40 × `a`, digests 64 × `b`, ids are `a1` × 16. `measure.py` reruns totals with every hex string of 32 or more chars swapped for sha256-derived hex of the same length.

| tokenizer | 11: ABIOS fixture → random | 11: JSON fixture → random | 200: ABIOS fixture → random | 200: JSON fixture → random | 200: ABIOS / JSON, random |
| --- | ---: | ---: | ---: | ---: | ---: |
| o200k_base | 761 → 769 | 738 → 721 | 7791 → 10445 | 28498 → 30738 | 0.340 |
| cl100k_base | 752 → 760 | 737 → 720 | 7855 → 10509 | 28700 → 30940 | 0.340 |
| llama3 | 752 → 760 | 737 → 720 | 7855 → 10509 | 28700 → 30940 | 0.340 |
| qwen3 | 758 → 869 | 738 → 872 | 7997 → 12518 | 28835 → 37710 | 0.332 |
| mistral_nemo | 870 → 928 | 848 → 909 | 10085 → 12642 | 34515 → 39105 | 0.323 |
| gemma3 | 834 → 930 | 808 → 911 | 8194 → 12700 | 31744 → 38429 | 0.330 |
| deepseek_v3 | 831 → 826 | 802 → 757 | 8148 → 10789 | 31630 → 31835 | 0.339 |

## Findings

1. **Hex ids are the largest cost in JSON and the clearest tokenizer gap.** They are 48–53% of JSON tokens on `lifecycle_200`. Random 32-char hex costs 17–18 tokens in o200k, cl100k, Llama 3 and DeepSeek V3, and 27–28 in Qwen3, Mistral Nemo and Gemma 3, which split digits one by one. The fixture ids (`a1` × 16) cost 32 tokens in every tokenizer, one per character, which is worse than random hex.
2. **Setup decides short sessions.** On `lifecycle_11`, setup is 525 of 761 o200k tokens. The ids block alone is about 183 tokens at 1.32 chars per token. ABIOS messages are 236 tokens against 738 for JSON, but ABIOS still loses by 23. With random hex in Qwen3, ABIOS wins by 3 (869 vs 872). In o200k it loses by 48 (769 vs 721).
3. **Quoted JSON lists inside fields cost more than their content.** Quotes, escapes and JSON brackets are 11.9% of ABIOS tokens on `lifecycle_200` in o200k (7.9% + 4.0%), at 1.5–2.3 chars per token. ` "[\"CI not executed in transcript.\"]"` costs 10–11 tokens for 5 words. The closing run `\"]" "` costs 3.5 tokens and occurs 135 times.
4. **Numbers and refs pay a digit tax.** The only numbers here are the `tool_version` values. ` 3.14` costs 4 tokens (5 in Qwen3, Nemo, Gemma) and occurs 130 times, about 5–7% of ABIOS tokens. ` #1` costs 2 tokens everywhere. ` #12` and above cost 3–4 tokens in Qwen3, Nemo and Gemma, and 2 in the rest. This fixture only reaches `#6`, so longer sessions would pay more in those three.
5. **Lexicon command words are already cheap.** They are 2.1–2.6% of ABIOS tokens on `lifecycle_200`, at about 4.2 chars per token. Making each one a single token saves 0 tokens in five tokenizers and 5 in Mistral Nemo and DeepSeek V3. The format tag `E` is a separate token in every case, so `Etest` is 2 tokens. A fused tag+command token would save one token per line. That is 200 on `lifecycle_200`, 2.6% of the o200k total, but it was not part of the counterfactual.
6. **Counterfactual savings are small.** Treating lexicon words and refs as single tokens (B) saves 244–274 tokens on `lifecycle_200`, 2.5–3.3% of the total. Adding one token per structural run (C) saves 494–664, 5.0–8.1%. On `lifecycle_11` both save between −5 and +2 tokens. Isolating separators (A) costs tokens in every case.
7. **Fixture hex flatters ABIOS on long sessions.** With random hex, the ABIOS/JSON ratio on `lifecycle_200` goes from 0.26–0.29 to 0.32–0.34. The 40-char revision is under the 48-byte ref threshold, so it is written in full on about 150 lines. Random hex makes each copy cost 19 tokens instead of 7 in o200k.
8. **Llama 3.1 and cl100k give identical counts on every stream.** Llama 3's vocabulary extends cl100k. On this ASCII traffic the extra tokens change no count.

## Limits

- The traffic is synthetic. `lifecycle_200` has only 38 distinct lines, and refs only reach `#6`. The hex is low-entropy (see the sensitivity table).
- Fractional attribution splits a token evenly over its characters. Class shares for separators and quotes depend on that choice. The counterfactual does not use it.
- The counterfactual counts tokens only. It ignores that a model would need training to use new tokens, and the cuts between pieces add their own cost, which is why both "vs stock" and "vs segmented" are shown.
- The "ideal" token counts in the units table are a simple rule, not an information measure.
- Class boundaries are judgment calls. `tool` (`python`) counts as prose, not enum. Version strings such as `3.14` count as numbers in both encodings. Hex counts as an id only at 16 or more chars.
- The counterfactual changes only the ABIOS message stream. Setup and JSON stay stock.
- Gemma and Llama come from ungated mirrors. Their tokenizer.json hashes are recorded but were not compared with the gated originals.
- Chat templates and special tokens are left out. A real request adds those on top.
- The abios source is commit `6da3a64`, exported with `git archive`. The local abios checkout was at `31d90a3` and was not changed.

## Files

- `extract_streams.py`: builds `data/streams.json` from abios.
- `measure.py`: tokenizes, checks the sanity numbers, writes `results/`.
- `requirements.txt`: pinned versions.
- `data/streams.json`: the three encodings of each task, plus setup text.
- `results/totals.csv`, `crossover.csv`, `classes.csv`, `units.csv`, `counterfactual.csv`, `random_hex.csv`, `probes.csv`, `results.json`: raw results.
- `results/tables.md`: every table, generated.
