# tokenpage

Tokenizer research for agent-to-agent traffic.

A DOS code page was the table that mapped bytes to characters. A tokenizer is the modern version: the table that maps text to the tokens a model reads and bills for. This repo asks what that table should look like when most of the text is agents talking to agents.

## Why

[ABIOS](https://github.com/Moissanite-Labs/abios) compresses agent messages as far as today's tokenizers allow. Past that point the vocabulary is the bottleneck: dense symbols, ids and structured fields split into many tokens because the vocabulary was trained on human prose and code. We can't change the tokenizers of hosted models. We can change the tokenizers of open-weight models, measure the result, and publish evidence the labs can act on.

## What lives here

- Measurements of how current vocabularies (o200k, Llama, Qwen and others) tokenize agent traffic.
- Experiments on open-weight models: reserved protocol tokens, merges tuned for structured data, session-adaptive vocabularies.
- Reproducible benchmarks, with the model, tokenizer, date, accuracy, tokens, latency and cost of every run.
- Proposals written for model labs, each backed by results from this repo.
- Cross-language density: how much of the character savings of dense languages such as Chinese survives each tokenizer, and what would keep it.
- A symbolic agent language, where one symbol is one common idea, with a harness translator to and from human language. Tested as a hypothesis.

See [CHARTER.md](CHARTER.md) for scope, questions and success criteria.

## Related

- [abios](https://github.com/Moissanite-Labs/abios): the message protocol this work extends.
- [nullmodem](https://github.com/Moissanite-Labs/nullmodem): agent communication below text, for models the harness controls.

## License

MIT
