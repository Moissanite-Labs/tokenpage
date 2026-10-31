#!/usr/bin/env python3
"""Study 01: where public tokenizers waste tokens on ABIOS benchmark traffic.

Reads data/streams.json (exported from abios by extract_streams.py) and
writes results/. Measurement only: no model calls, no paid APIs.

    python measure.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAMS = HERE / "data" / "streams.json"
OUT = HERE / "results"

# name -> (kind, source, revision). HF revisions are pinned commit SHAs.
TOKENIZERS = {
    "o200k_base": ("tiktoken", "o200k_base", None),
    "cl100k_base": ("tiktoken", "cl100k_base", None),
    "llama3": ("hf", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
    "qwen3": ("hf", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "mistral_nemo": ("hf", "mistralai/Mistral-Nemo-Instruct-2407", "04d8a90549d23fc6bd7f642064003592df51e9b3"),
    "gemma3": ("hf", "unsloth/gemma-3-1b-it", "5b11413a10db4e486ef16a20101fd028f8f2499c"),
    "deepseek_v3": ("hf", "deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb"),
}

# Known o200k_base totals from abios docs/abios-bench.md at 6da3a64
# (POST + minimal setup + messages for ABIOS refs; messages only for JSON).
SANITY = {
    "lifecycle_11": {"abios_refs": 761, "json_compact": 738},
    "lifecycle_40": {"abios_refs": 1983, "json_compact": 5570},
    "lifecycle_200": {"abios_refs": 7791, "json_compact": 28498},
}

ENCODINGS = ("abios_refs", "json_compact", "english_lossless")
ENUM_FIELDS = {"kind", "status", "decision", "priority", "capabilities", "action"}
HANDLES = {"p", "t", "a", "h", "f"}
REF_RE = re.compile(r"^#\d+$")
HEX_RE = re.compile(r"^[0-9a-f]{8,}$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)*$")
WORD_RE = re.compile(r"\w+")


# ---------------------------------------------------------------- tokenizers
class Tok:
    def __init__(self, name: str):
        kind, source, rev = TOKENIZERS[name]
        self.name, self.kind, self.source, self.revision = name, kind, source, rev
        if kind == "tiktoken":
            import tiktoken

            self._enc = tiktoken.get_encoding(source)
            self.file_sha256 = None
            self.vocab_size = self._enc.n_vocab
            self.lib = f"tiktoken {version('tiktoken')}"
        else:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            path = hf_hub_download(source, "tokenizer.json", revision=rev)
            self.file_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self._enc = Tokenizer.from_file(path)
            self.vocab_size = self._enc.get_vocab_size()
            self.lib = f"tokenizers {version('tokenizers')}"
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text == "":
            return 0
        hit = self._cache.get(text)
        if hit is None:
            if self.kind == "tiktoken":
                hit = len(self._enc.encode(text, disallowed_special=()))
            else:
                hit = len(self._enc.encode(text, add_special_tokens=False).ids)
            if len(text) < 400:
                self._cache[text] = hit
        return hit

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Char span of every token, in order."""
        if self.kind == "tiktoken":
            ids = self._enc.encode(text, disallowed_special=())
            _, starts = self._enc.decode_with_offsets(ids)
            ends = starts[1:] + [len(text)]
            return list(zip(starts, ends))
        return list(self._enc.encode(text, add_special_tokens=False).offsets)

    def char_weights(self, text: str) -> tuple[list[float], int, int]:
        """Per-char share of a token, so a class sum gives fractional tokens."""
        w = [0.0] * len(text)
        spans = self.spans(text)
        for s, e in spans:
            if e > s:
                for i in range(s, e):
                    w[i] += 1.0 / (e - s)
            elif len(text):
                w[min(s, len(text) - 1)] += 1.0
        uncovered = sum(1 for x in w if x == 0.0)
        return w, len(spans), uncovered


# ------------------------------------------------------------- ABIOS parsing
def parse_setup_meta(setup: str) -> tuple[str, dict[str, str], dict[str, list[str]]]:
    lines = setup.split("\n")
    lex = json.loads(lines[lines.index("lexicon:") + 1])
    schema = json.loads(lines[lines.index("schema:") + 1])
    tag = lex["format_tags"]["refs"]
    word_to_cmd = {w: c for c, w in lex["concepts"]["cmd"].items()}
    return tag, word_to_cmd, schema


def classify_setup(setup: str) -> list[str]:
    cls = []
    block = "setup_ack"
    for idx, line in enumerate(setup.split("\n")):
        if idx == 0:
            block = "setup_ack"
        elif line in ("lexicon:", "schema:", "ids:", "defaults:"):
            block = "setup_" + line[:-1]
        elif idx in (1, 2):
            block = "setup_headers"
        cls.extend([block] * len(line))
        cls.append(block)  # the newline
    return cls[: len(setup)]


def _json_struct_classes(chars: list[str], item_cls: str) -> list[str]:
    """Classify chars of a JSON value: structure vs string content."""
    out, in_str, esc = [], False, False
    for ch in chars:
        if in_str:
            if esc:
                out.append("json_struct"); esc = False
            elif ch == "\\":
                out.append("json_struct"); esc = True
            elif ch == '"':
                out.append("json_struct"); in_str = False
            else:
                out.append(item_cls)
        else:
            if ch == '"':
                in_str = True
            out.append("json_struct" if not ch.isalnum() else "number")
    return out


def _bare_class(text: str, field: str) -> str:
    if REF_RE.match(text):
        return "content_ref"
    if text in HANDLES:
        return "session_handle"
    if HEX_RE.match(text):
        return "hex_id"
    if NUM_RE.match(text) or text == "-":
        return "number"
    if field in ENUM_FIELDS:
        return "enum_value"
    return "payload_prose"


def _ideal(text: str, kind: str) -> int:
    """Information floor in tokens (see README, 'Worst-splitting patterns')."""
    if kind in ("hex_id",):
        return math.ceil(len(text.strip(' "')) / 4)
    if kind in ("payload_prose", "quoted_prose", "quoted_json", "bare_json"):
        return max(1, len(WORD_RE.findall(text)))
    return 1


def parse_abios_line(line: str, tag: str, word_to_cmd: dict, schema: dict):
    """Return (char classes, units). Units: (text, kind, start, end)."""
    if not line.startswith(tag):
        raise ValueError(f"line without tag: {line[:40]}")
    cls = ["tag"] * len(tag)
    sp = line.index(" ", len(tag)) if " " in line else len(line)
    word = line[len(tag):sp]
    cmd = word_to_cmd[word]
    cls += ["cmd_word"] * len(word)
    units = [(line[:sp], "tag+cmd", 0, sp)]
    fields = schema[cmd]
    i, fi = sp, 0
    while i < len(line):
        assert line[i] == " ", (line, i)
        cls.append("field_sep")
        i += 1
        start = i
        field = fields[fi]
        fi += 1
        if i < len(line) and line[i] == '"':
            j = i + 1
            raw: list[tuple[int, str]] = []  # (raw index, unescaped char)
            esc_idx: list[int] = []
            while line[j] != '"':
                if line[j] == "\\":
                    esc_idx.append(j)
                    nxt = line[j + 1]
                    raw.append((j + 1, {"n": "\n", "r": "\r"}.get(nxt, nxt)))
                    j += 2
                else:
                    raw.append((j, line[j]))
                    j += 1
            end = j + 1
            content = "".join(c for _, c in raw)
            sub = ["quote_escape"] * (end - start)
            is_json = False
            if content[:1] in "[{":
                try:
                    json.loads(content)
                    is_json = True
                except ValueError:
                    pass
            if is_json:
                item = "enum_value" if field in ENUM_FIELDS else "payload_prose"
                ccls = _json_struct_classes([c for _, c in raw], item)
                kind = "quoted_json"
            else:
                base = _bare_class(content, field) if content else "quote_escape"
                if base in ("number", "session_handle", "content_ref"):
                    base = "payload_prose"
                ccls = [base] * len(raw)
                kind = "quoted_prose" if base == "payload_prose" else base
            for (ri, _), c in zip(raw, ccls):
                sub[ri - start] = c
            cls += sub
            i = end
        else:
            j = line.find(" ", i)
            end = len(line) if j == -1 else j
            text = line[i:end]
            if text[:1] in "[{":
                item = "enum_value" if field in ENUM_FIELDS else "payload_prose"
                cls += _json_struct_classes(list(text), item)
                kind = "bare_json"
            else:
                kind = _bare_class(text, field)
                cls += [kind] * len(text)
            i = end
        units.append((" " + line[start:end], kind, start - 1, end))
    assert fi == len(fields), f"arity mismatch {cmd}"
    assert len(cls) == len(line)
    return cls, units


def classify_abios_messages(lines: list[str], setup: str):
    tag, w2c, schema = parse_setup_meta(setup)
    cls, units, off = [], [], 0
    for n, line in enumerate(lines):
        if n:
            cls.append("line_sep")
            off += 1
        c, u = parse_abios_line(line, tag, w2c, schema)
        cls += c
        units += [(t, k, s + off, e + off) for t, k, s, e in u]
        off += len(line)
    return cls, units


# -------------------------------------------------------------- JSON parsing
def classify_json_line(line: str, base: int = 0):
    cls = [""] * len(line)
    units = []
    stack: list[list] = []  # [container, expecting_key, current_key]
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            j = i + 1
            while line[j] != '"':
                j += 2 if line[j] == "\\" else 1
            text = line[i + 1 : j]
            top = stack[-1] if stack else None
            if top and top[0] == "{" and top[1]:
                for k in range(i, j + 1):
                    cls[k] = "key"
                top[2] = json.loads(line[i : j + 1])
                units.append((line[i : j + 2], "key", base + i, base + j + 2))  # include ':'
            else:
                key = next((s[2] for s in reversed(stack) if s[0] == "{"), "")
                if key in ENUM_FIELDS:
                    vc = "enum_value"
                elif HEX_RE.match(text) and len(text) >= 16:
                    vc = "hex_id"
                else:
                    vc = "payload_prose"
                cls[i] = cls[j] = "punct"
                for k in range(i + 1, j):
                    cls[k] = "punct" if line[k] == "\\" else vc
                if top and top[0] == "{":
                    kind = {"payload_prose": "quoted_prose"}.get(vc, vc)
                    units.append((line[i : j + 1], kind, base + i, base + j + 1))
            i = j + 1
            continue
        if ch in "{[":
            cls[i] = "punct"
            stack.append([ch, ch == "{", ""])
            if ch == "[" and len(stack) > 1 and stack[-2][0] == "{":
                close = _match(line, i)
                units.append((line[i : close + 1], "bare_json", base + i, base + close + 1))
        elif ch in "}]":
            cls[i] = "punct"
            stack.pop()
        elif ch == ":":
            cls[i] = "punct"
            stack[-1][1] = False
        elif ch == ",":
            cls[i] = "punct"
            if stack[-1][0] == "{":
                stack[-1][1] = True
        else:
            m = re.match(r"-?\d+(\.\d+)?|true|false|null", line[i:])
            for k in range(i, i + m.end()):
                cls[k] = "number"
            units.append((m.group(0), "number", base + i, base + i + m.end()))
            i += m.end()
            continue
        i += 1
    assert all(cls), line
    return cls, units


def _match(line: str, i: int) -> int:
    depth, in_str, esc = 0, False, False
    for k in range(i, len(line)):
        ch = line[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError("unbalanced")


def classify_json_messages(lines: list[str]):
    cls, units, off = [], [], 0
    for n, line in enumerate(lines):
        if n:
            cls.append("line_sep")
            off += 1
        c, u = classify_json_line(line, off)
        cls += c
        units += u
        off += len(line)
    return cls, units


# -------------------------------------------------------------- measurements
def crossover(tok: Tok, setup_n: int, enc: list[str], base: list[str]):
    for i in range(1, len(enc) + 1):
        if setup_n + tok.count("\n".join(enc[:i])) < tok.count("\n".join(base[:i])):
            return i
    return None


def class_table(tok: Tok, text: str, cls: list[str]):
    w, ntok, uncovered = tok.char_weights(text)
    chars, toks = defaultdict(int), defaultdict(float)
    for c, x in zip(cls, w):
        chars[c] += 1
        toks[c] += x
    return chars, toks, ntok, uncovered, w


def counterfactual(tok: Tok, lines: list[str], setup: str):
    """Message-stream token counts under hypothetical vocabularies.

    A (isolated): tag, cmd word, each field separator, each line separator
      and each #n ref (n<1000) is one token; everything else is tokenized
      by the stock tokenizer as isolated pieces.
    B (space-prefixed): tag, cmd word and ' #n' (separator + ref) are one
      token each; other fields keep their leading space, tokenized stock.
    C (B + structure): as B, and every run of quoting/escaping/JSON
      punctuation inside a field (with its leading space) is one token.
    Each variant also reports a 'segmented stock' count: the same pieces,
    all tokenized by the stock tokenizer, to separate the vocabulary
    effect from the effect of cutting the stream into pieces.
    """
    tag, w2c, schema = parse_setup_meta(setup)
    res = {v: {"cf": 0, "seg": 0} for v in ("A", "B", "C")}
    comp = {"B_lexicon_only": 0, "B_refs_only": 0}
    for n, line in enumerate(lines):
        cls, units = parse_abios_line(line, tag, w2c, schema)
        if n:
            for v in res:
                res[v]["cf"] += 1
                res[v]["seg"] += tok.count("\n")
            comp["B_lexicon_only"] += tok.count("\n")
            comp["B_refs_only"] += tok.count("\n")
        head = units[0][0]
        word = head[len(tag):]
        t_tag, t_word = tok.count(tag), tok.count(word)
        for v in res:
            res[v]["cf"] += 2
            res[v]["seg"] += t_tag + t_word
        comp["B_lexicon_only"] += 2
        comp["B_refs_only"] += t_tag + t_word
        for text, kind, s, e in units[1:]:
            body = text[1:]
            is_ref = kind == "content_ref" and int(body[1:]) < 1000
            # A
            res["A"]["seg"] += tok.count(" ") + tok.count(body)
            res["A"]["cf"] += 1 + (1 if is_ref else tok.count(body))
            # B
            stock = tok.count(text)
            res["B"]["seg"] += stock
            res["B"]["cf"] += 1 if is_ref else stock
            comp["B_lexicon_only"] += stock
            comp["B_refs_only"] += 1 if is_ref else stock
            # C: split field into structural runs and content runs
            if is_ref:
                res["C"]["seg"] += stock
                res["C"]["cf"] += 1
                continue
            runs, cur, cur_struct = [], "", None
            for ch, c in zip(text, ["field_sep"] + cls[s + 1 : e]):
                st = c in ("field_sep", "quote_escape", "json_struct")
                if cur and st != cur_struct:
                    runs.append((cur, cur_struct))
                    cur = ""
                cur += ch
                cur_struct = st
            runs.append((cur, cur_struct))
            # a lone leading space stays glued to the next content run
            if runs[0] == (" ", True) and len(runs) > 1:
                runs = [(" " + runs[1][0], runs[1][1])] + runs[2:]
            for r, st in runs:
                res["C"]["seg"] += tok.count(r)
                res["C"]["cf"] += 1 if st else tok.count(r)
    out = {}
    for v, d in res.items():
        out[v] = {"messages_cf": d["cf"], "messages_segmented_stock": d["seg"]}
    out.update({k: {"messages_cf": v} for k, v in comp.items()})
    return out


def realistic_hex(text: str) -> str:
    """Replace low-entropy fixture hex (>=32 chars) with sha256-derived hex."""
    def sub(m):
        s = m.group(0)
        h = ""
        k = 0
        while len(h) < len(s):
            h += hashlib.sha256(f"tokenpage-01:{s}:{k}".encode()).hexdigest()
            k += 1
        return h[: len(s)]

    return re.sub(r"(?<![0-9a-f])[0-9a-f]{32,}(?![0-9a-f])", sub, text)


PROBES = {
    "ref #1 (with leading space)": " #1",
    "ref #12": " #12",
    "ref #123": " #123",
    "ref #999": " #999",
    "handle ' t'": " t",
    "fixture revision (40 x 'a')": " " + "a" * 40,
    "fixture digest (64 x 'b')": " " + "b" * 64,
    "fixture id (16 x 'a1')": " " + "a1" * 16,
    "random sha1 hex (40)": " " + hashlib.sha1(b"tokenpage").hexdigest(),
    "random sha256 hex (64)": " " + hashlib.sha256(b"tokenpage").hexdigest(),
    "random uuid4-style hex (32)": " " + hashlib.md5(b"tokenpage").hexdigest(),
    "JSON-in-quotes opener ' \"[\\\"'": ' "[\\"',
    "JSON-in-quotes closer '\\\"]\"'": '\\"]"',
}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


CLASS_LABELS = {
    "payload_prose": "payload prose",
    "hex_id": "hex ids / digests",
    "number": "numbers (e.g. 3.14)",
    "session_handle": "session handles p/t/a/h/f",
    "content_ref": "content refs #n",
    "field_sep": "field separator (space)",
    "line_sep": "line separator (newline)",
    "quote_escape": "quotes and backslash escapes",
    "json_struct": "JSON structure inside a field",
    "enum_value": "enum values (kind, status, ...)",
    "cmd_word": "lexicon command word",
    "tag": "format tag E",
    "setup_ack": "setup: POST ack line",
    "setup_headers": "setup: profile header",
    "setup_lexicon": "setup: lexicon block",
    "setup_schema": "setup: schema block",
    "setup_ids": "setup: ids block",
    "setup_defaults": "setup: defaults block",
    "key": "keys (with quotes)",
    "punct": "punctuation and value quotes",
}
TASK_ORDER = ("lifecycle_11", "lifecycle_40", "lifecycle_200")


def _md(header: list[str], rows: list[list], align: str | None = None) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join(align or ["---"] + ["---:"] * (len(header) - 1)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def render_tables(res: dict, unit_rows: list[dict]) -> str:
    names = list(res["tokenizers"])
    tot = {(r["task"], r["tokenizer"], r["encoding"]): r for r in res["totals"]}
    parts = ["# Study 01 tables (generated by measure.py; do not edit)\n"]

    parts.append("## Totals (tokens; ABIOS = POST + minimal setup + messages)\n")
    rows = []
    for t in TASK_ORDER:
        for n in names:
            a = tot[(t, n, "abios_refs")]["total"]
            j = tot[(t, n, "json_compact")]["total"]
            e = tot[(t, n, "english_lossless")]["total"]
            rows.append([t, n, a, j, e, f"{a / j:.3f}"])
    parts.append(_md(["task", "tokenizer", "ABIOS refs", "compact JSON", "English lossless", "ABIOS / JSON"], rows, ["---", "---"] + ["---:"] * 4))

    parts.append("\n## Crossover (first message count where cumulative ABIOS total < baseline)\n")
    rows = [[r["task"], r["tokenizer"], r["crossover_vs_json_msgs"] or "never", r["crossover_vs_english_msgs"] or "never"] for r in res["crossover"]]
    parts.append(_md(["task", "tokenizer", "vs compact JSON", "vs English lossless"], rows, ["---", "---", "---:", "---:"]))

    for enc in ("abios_refs", "json_compact"):
        for t in ("lifecycle_11", "lifecycle_200"):
            parts.append(f"\n## Waste breakdown: {enc}, {t} (share of tokens / chars per token)\n")
            cl = [r for r in res["classes"] if r["task"] == t and r["encoding"] == enc]
            order = []
            for r in sorted((r for r in cl if r["tokenizer"] == "o200k_base"), key=lambda r: -r["tokens"]):
                order.append(r["class"])
            idx = {(r["class"], r["tokenizer"]): r for r in cl}
            rows = []
            for c in order:
                row = [CLASS_LABELS.get(c, c), idx[(c, "o200k_base")]["chars"]]
                for n in names:
                    r = idx[(c, n)]
                    row.append(f"{100 * r['share']:.1f}% / {r['chars_per_token']}")
                rows.append(row)
            parts.append(_md(["class", "chars"] + names, rows, ["---"] + ["---:"] * (len(names) + 1)))

    parts.append("\n## Worst-splitting recurring units: abios_refs, lifecycle_200 (tokens per occurrence)\n")
    parts.append("Ranked by excess = occurrences x (mean tokens - ideal tokens). Structural runs overlap the quoted fields they sit in.\n\n")
    ur = [r for r in unit_rows if r["task"] == "lifecycle_200" and r["encoding"] == "abios_refs" and r["occurrences"] >= 2]
    ur.sort(key=lambda r: -r["excess_total"])
    rows = []
    for r in ur[:20]:
        u = r["unit"].replace("|", "\\|")
        rows.append([f"`{u}`", r["kind"], r["occurrences"], r["ideal_tokens"]] + [r[f"tok_{n}"] for n in names] + [r["excess_total"]])
    parts.append(_md(["unit", "kind", "occ", "ideal"] + names + ["excess total (mean)"], rows, ["---", "---"] + ["---:"] * (len(names) + 3)))

    parts.append("\n## Worst-splitting recurring units: json_compact, lifecycle_200 (top 10)\n")
    ur = [r for r in unit_rows if r["task"] == "lifecycle_200" and r["encoding"] == "json_compact" and r["occurrences"] >= 2]
    ur.sort(key=lambda r: -r["excess_total"])
    rows = []
    for r in ur[:10]:
        u = r["unit"] if len(r["unit"]) <= 48 else r["unit"][:45] + "..."
        rows.append([f"`{u}`", r["kind"], r["occurrences"], r["ideal_tokens"]] + [r[f"tok_{n}"] for n in names] + [r["excess_total"]])
    parts.append(_md(["unit", "kind", "occ", "ideal"] + names + ["excess total (mean)"], rows, ["---", "---"] + ["---:"] * (len(names) + 3)))

    parts.append("\n## Probes (tokens for one isolated string)\n")
    rows = [[p["probe"], f"`{p['text']}`"] + [p[n] for n in names] for p in res["probes"]]
    parts.append(_md(["probe", "text"] + names, rows, ["---", "---"] + ["---:"] * len(names)))

    parts.append("\n## Counterfactual vocabulary (estimate, not a trained tokenizer)\n")
    rows = []
    for r in res["counterfactual"]:
        if r["variant"] in ("A", "B", "C"):
            rows.append([r["task"], r["tokenizer"], r["variant"], r["total_stock"], r["total_cf"],
                         r["saving_vs_stock_tokens"], f"{r['saving_vs_stock_pct_of_total']}%",
                         r["saving_vs_segmented_tokens"], r["cf_total_vs_json_ratio"]])
    parts.append(_md(["task", "tokenizer", "variant", "total stock", "total CF", "saved vs stock", "saved % of total", "saved vs segmented stock", "CF total / JSON"], rows, ["---", "---", "---"] + ["---:"] * 6))

    parts.append("\n## Random-hex sensitivity (fixture hex >= 32 chars replaced by sha256-derived hex)\n")
    hx = {(r["task"], r["tokenizer"], r["encoding"]): r for r in res["random_hex"]}
    rows = []
    for t in TASK_ORDER:
        for n in names:
            a, j = hx[(t, n, "abios_refs")], hx[(t, n, "json_compact")]
            rows.append([t, n, a["total_fixture_hex"], a["total_random_hex"], j["total_fixture_hex"], j["total_random_hex"],
                         f"{a['total_random_hex'] / j['total_random_hex']:.3f}"])
    parts.append(_md(["task", "tokenizer", "ABIOS fixture", "ABIOS random", "JSON fixture", "JSON random", "ABIOS / JSON (random)"], rows, ["---", "---"] + ["---:"] * 5))
    return "".join(parts)


def main() -> int:
    data = json.loads(STREAMS.read_text())
    tasks = data["tasks"]
    OUT.mkdir(exist_ok=True)
    toks = {n: Tok(n) for n in TOKENIZERS}
    meta = {
        n: {
            "library": t.lib,
            "source": t.source,
            "revision": t.revision,
            "tokenizer_json_sha256": t.file_sha256,
            "vocab_size": t.vocab_size,
        }
        for n, t in toks.items()
    }

    totals, cross, classes, cf_rows, unit_rows, hex_rows, probe_rows = [], [], [], [], [], [], []
    sanity = []
    coverage = []

    for task in sorted(tasks, key=lambda t: tasks[t]["count"]):
        d = tasks[task]
        setup = d["abios_setup"]
        streams = {e: "\n".join(d[e]) for e in ENCODINGS}
        a_cls, a_units = classify_abios_messages(d["abios_refs"], setup)
        j_cls, j_units = classify_json_messages(d["json_compact"])
        s_cls = classify_setup(setup)
        unit_acc = {"abios_refs": defaultdict(dict), "json_compact": defaultdict(dict)}
        for name, tok in toks.items():
            setup_n = tok.count(setup)
            row_tot = {}
            for e in ENCODINGS:
                msgs = tok.count(streams[e])
                s = setup_n if e == "abios_refs" else 0
                row_tot[e] = s + msgs
                totals.append({"task": task, "tokenizer": name, "encoding": e,
                               "setup": s, "messages": msgs, "total": s + msgs})
            if name == "o200k_base":
                for e, want in SANITY[task].items():
                    sanity.append({"task": task, "encoding": e, "expected": want,
                                   "measured": row_tot[e], "match": want == row_tot[e]})
                assert setup_n == d["bench_o200k_post_minimal"], (task, setup_n)
            cross.append({
                "task": task, "tokenizer": name,
                "abios_total": row_tot["abios_refs"], "json_total": row_tot["json_compact"],
                "english_total": row_tot["english_lossless"],
                "abios_vs_json_ratio": round(row_tot["abios_refs"] / row_tot["json_compact"], 4),
                "crossover_vs_json_msgs": crossover(tok, setup_n, d["abios_refs"], d["json_compact"]),
                "crossover_vs_english_msgs": crossover(tok, setup_n, d["abios_refs"], d["english_lossless"]),
            })

            # waste breakdown
            enc_parts = {
                "abios_refs": [(setup, s_cls), (streams["abios_refs"], a_cls)],
                "json_compact": [(streams["json_compact"], j_cls)],
            }
            weights = {}
            for e, parts in enc_parts.items():
                chars, tks = defaultdict(int), defaultdict(float)
                total_tok = 0
                for text, cls in parts:
                    c, t, ntok, unc, w = class_table(tok, text, cls)
                    coverage.append({"task": task, "tokenizer": name, "encoding": e,
                                     "chars": len(text), "uncovered_chars": unc})
                    for k in c:
                        chars[k] += c[k]
                        tks[k] += t[k]
                    total_tok += ntok
                    weights[(e, text is setup)] = w
                for k in sorted(chars, key=lambda k: -tks[k]):
                    classes.append({
                        "task": task, "tokenizer": name, "encoding": e, "class": k,
                        "chars": chars[k], "tokens": round(tks[k], 2),
                        "share": round(tks[k] / total_tok, 4),
                        "chars_per_token": round(chars[k] / tks[k], 3) if tks[k] else None,
                    })
            # units (in-context fractional tokens)
            for e, units in (("abios_refs", a_units), ("json_compact", j_units)):
                w = weights[(e, False)]
                acc = unit_acc[e]
                for text, kind, s, en in units:
                    key = (text, kind)
                    slot = acc[key].setdefault(name, [0, 0.0])
                    slot[0] += 1
                    slot[1] += sum(w[s:en])
                # structural runs in ABIOS (overlap with field units, reported apart)
                if e == "abios_refs":
                    run_s = None
                    st_cls = ("field_sep", "quote_escape", "json_struct")
                    for idx in range(len(a_cls) + 1):
                        st = idx < len(a_cls) and a_cls[idx] in st_cls
                        if st and run_s is None:
                            run_s = idx
                        elif not st and run_s is not None:
                            if idx - run_s >= 2:
                                text = streams[e][run_s:idx]
                                slot = acc[(text, "structural_run")].setdefault(name, [0, 0.0])
                                slot[0] += 1
                                slot[1] += sum(w[run_s:idx])
                            run_s = None

            # counterfactual
            cf = counterfactual(tok, d["abios_refs"], setup)
            base_msgs = tok.count(streams["abios_refs"])
            for v, r in cf.items():
                msgs_cf = r["messages_cf"]
                seg = r.get("messages_segmented_stock")
                cf_rows.append({
                    "task": task, "tokenizer": name, "variant": v,
                    "setup": setup_n, "messages_stock": base_msgs,
                    "messages_segmented_stock": seg, "messages_cf": msgs_cf,
                    "total_stock": setup_n + base_msgs, "total_cf": setup_n + msgs_cf,
                    "saving_vs_stock_tokens": base_msgs - msgs_cf,
                    "saving_vs_stock_pct_of_total": round(100 * (base_msgs - msgs_cf) / (setup_n + base_msgs), 2),
                    "saving_vs_segmented_tokens": (seg - msgs_cf) if seg is not None else None,
                    "cf_total_vs_json_ratio": round((setup_n + msgs_cf) / row_tot["json_compact"], 4),
                })

            # realistic hex sensitivity
            for e in ENCODINGS:
                s = tok.count(realistic_hex(setup)) if e == "abios_refs" else 0
                m = tok.count(realistic_hex(streams[e]))
                hex_rows.append({"task": task, "tokenizer": name, "encoding": e,
                                 "total_fixture_hex": row_tot[e], "total_random_hex": s + m,
                                 "delta": s + m - row_tot[e]})

        for e, acc in unit_acc.items():
            for (text, kind), per in acc.items():
                occ = next(iter(per.values()))[0]
                ideal = _ideal(text, kind)
                row = {"task": task, "encoding": e, "unit": text, "kind": kind,
                       "occurrences": occ, "chars": len(text), "ideal_tokens": ideal}
                per_occ = []
                for name in toks:
                    v = per[name][1] / per[name][0]
                    row[f"tok_{name}"] = round(v, 2)
                    per_occ.append(v)
                mean = sum(per_occ) / len(per_occ)
                row["tok_mean"] = round(mean, 2)
                row["excess_per_occ"] = round(mean - ideal, 2)
                row["excess_total"] = round(occ * (mean - ideal), 1)
                row["tokens_per_ideal"] = round(mean / ideal, 2)
                unit_rows.append(row)

    for label, text in PROBES.items():
        r = {"probe": label, "text": text}
        for name, tok in toks.items():
            r[name] = tok.count(text)
        probe_rows.append(r)

    unit_rows.sort(key=lambda r: (r["task"], r["encoding"], -r["excess_total"]))
    write_csv(OUT / "totals.csv", totals)
    write_csv(OUT / "crossover.csv", cross)
    write_csv(OUT / "classes.csv", classes)
    write_csv(OUT / "units.csv", unit_rows)
    write_csv(OUT / "counterfactual.csv", cf_rows)
    write_csv(OUT / "random_hex.csv", hex_rows)
    write_csv(OUT / "probes.csv", probe_rows)
    result = {
        "study": "01-tokenizer-waste",
        "streams_sha256": hashlib.sha256(STREAMS.read_bytes()).hexdigest(),
        "abios_commit": data["abios_commit_seen"],
        "tokenizers": meta,
        "sanity_o200k": sanity,
        "offset_coverage": coverage,
        "totals": totals,
        "crossover": cross,
        "classes": classes,
        "counterfactual": cf_rows,
        "random_hex": hex_rows,
        "probes": probe_rows,
    }
    (OUT / "tables.md").write_text(render_tables(result, unit_rows), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    ok = all(s["match"] for s in sanity)
    for s in sanity:
        print(f"sanity {s['task']:14} {s['encoding']:13} expected {s['expected']:6} measured {s['measured']:6} {'OK' if s['match'] else 'MISMATCH'}")
    bad = [c for c in coverage if c["uncovered_chars"]]
    if bad:
        print(f"warning: {len(bad)} streams have chars not covered by token offsets")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
