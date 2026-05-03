# tl-corpus-v1

`tl-corpus-v1` is the single JSON/JSONL corpus schema used by this project.
Pretraining, SFT, and future preference tuning use the same record shape.

One JSON object is one independent training record:

```json
{
  "schema": "tl-corpus-v1",
  "id": "example-000001",
  "source": "dataset-or-file-name",
  "task": "lm",
  "segments": [
    {"role": "text", "content": "plain text for language-model pretraining", "train": true}
  ],
  "meta": {}
}
```

For SFT/chat records, `segments` contains the canonical conversation. Context
segments usually have `train=false`; assistant targets usually have
`train=true`.

```json
{
  "schema": "tl-corpus-v1",
  "id": "chat-000001",
  "source": "daily_dialog",
  "task": "sft",
  "segments": [
    {"role": "user", "content": "hello", "train": false},
    {"role": "assistant", "content": "Hi. How can I help?", "train": true}
  ],
  "meta": {}
}
```

For preference tuning, keep the chosen/positive branch in `segments` and put
the rejected/negative branch in optional `rejected_segments`.

```json
{
  "schema": "tl-corpus-v1",
  "id": "pref-000001",
  "source": "preference_dataset",
  "task": "preference",
  "segments": [
    {"role": "user", "content": "Explain attention briefly.", "train": false},
    {"role": "assistant", "content": "Attention lets each token read relevant earlier tokens.", "train": true}
  ],
  "rejected_segments": [
    {"role": "user", "content": "Explain attention briefly.", "train": false},
    {"role": "assistant", "content": "I do not know.", "train": true}
  ],
  "meta": {}
}
```

Fields:

- `schema`: always `tl-corpus-v1`.
- `id`: stable record id.
- `source`: dataset or file name.
- `task`: optional hint, normalized to `lm`, `sft`, or `preference`.
- `segments`: required positive/canonical text branch.
- `rejected_segments`: optional negative branch for preference training.
- `meta`: optional source metadata.

Current LM/SFT training reads only `segments`. Preference trainers can use
`rejected_segments` later without changing tokenizer ids, model checkpoints, or
the base corpus format.
