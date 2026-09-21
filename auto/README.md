# autodiag

OBD-II + service-manual retrieval + Claude, for engine troubleshooting.
Reads your car (or a recorded fixture), pulls the relevant pages out of the
manuals you have ingested, and returns ranked causes, the next tests to run,
and the evidence behind each claim. Design: [DESIGN.md](DESIGN.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ant auth login          # or: export ANTHROPIC_API_KEY=...
```

## Quick start (no car needed)

```bash
autodiag demo                 # ingest seed knowledge, diagnose the P0301 fixture
autodiag demo --offline       # same, but the rules-only report with no API call
```

## With a car

```bash
autodiag scan --port /dev/tty.OBDII         # writes snapshot.json
autodiag ingest ~/manuals/accord-2014/      # PDFs, Markdown, text
autodiag diagnose --snapshot snapshot.json \
    --symptoms "rough idle when cold, shakes at stoplights" \
    --audio idle.wav --image plug.jpg
```

Omit `--port` to auto-detect the adapter. Use `--slow` for flaky ELM327 clones.

## Offline

| Command | Behavior |
|---|---|
| `diagnose --offline` | Codes, derived flags, readiness, and matching manual excerpts. No API call. |
| `diagnose --queue` | Do everything except the API call; save the job under `~/.autodiag/queue/`. |
| `diagnose --drain` | Send queued jobs, write `diagnosis.json` and `report.txt` next to each. |
| `queue-list` | Show pending, failed, and completed jobs. |

If the API host is unreachable, `diagnose` falls back to the offline report automatically.

## Other commands

```bash
autodiag kb                          # list ingested documents
autodiag search "fuel trim vacuum leak"   # inspect what retrieval returns
autodiag diagnose --fixture fixtures/p0171_lean.json --no-vin --strict
```

## Configuration

Environment variables: `AUTODIAG_HOME` (default `~/.autodiag`), `AUTODIAG_KB`,
`AUTODIAG_QUEUE`, `AUTODIAG_REPORTS`, `AUTODIAG_MODEL` (default `claude-opus-5`),
`AUTODIAG_EFFORT` (default `high`), `AUTODIAG_MAX_TOKENS`.

### Local model (Ollama)

The reasoning step can run on a local model instead of Claude. Install
[Ollama](https://ollama.com), pull a model, and pick the backend:

```bash
ollama pull qwen2.5:14b                      # or llama3.1:8b, gemma3:12b, ...
autodiag demo --backend ollama               # one run
export AUTODIAG_BACKEND=ollama               # make it the default
export AUTODIAG_LOCAL_MODEL=qwen2.5:14b
export AUTODIAG_LOCAL_FALLBACK=1             # use Ollama automatically when the Claude host is unreachable
```

Ollama's schema-constrained output is used, and one repair round-trip runs if
the JSON still fails validation. Expect it to be slower and to cite manual
chunks less reliably than Claude; the report is labeled when a local model
produced it. Vision-capable local models (llava, gemma3, qwen-vl) receive
`--image` inputs; text-only models ignore them. `OLLAMA_HOST`,
`AUTODIAG_LOCAL_NUM_CTX` (default 16384) and `AUTODIAG_LOCAL_TIMEOUT` (default
600 s) tune the connection.

### Through a gateway

To route through a proxy that speaks the Anthropic Messages API (LiteLLM, an
internal router) and uses prefixed model names:

```bash
export ANTHROPIC_BASE_URL=https://your-gateway.example.com
export ANTHROPIC_API_KEY=<gateway key>
export AUTODIAG_MODEL=azure/anthropic/claude-opus-5
export AUTODIAG_FALLBACKS=0    # drop the refusal-fallback beta header if the gateway rejects it
```

The connectivity probe follows `ANTHROPIC_BASE_URL`. Structured output
(`output_config`) must pass through the gateway unchanged; if the gateway
strips it, the response will not parse and the run fails loudly rather than
returning prose.

## Tests

```bash
pytest
```

Tests cover OBD parsing, ingestion and chunking, retrieval ranking, flag
rules, citation validation, the offline report, the queue round-trip (with a
fake API client), both model backends (with fake Claude and Ollama servers),
and audio feature extraction. No network is needed.

## Layout

```
autodiag/
  models.py      pydantic models; Diagnosis doubles as the output JSON schema
  obd/           python-OBD reader and JSON fixture simulator
  knowledge/     SQLite FTS5 store, ingest, deterministic queries + RRF retriever
  reasoning/     flags, prompt, backends (Claude, Ollama), citation validation, offline report, queue
  media/         audio features (numpy), video frame sampling (ffmpeg)
  cli.py
knowledge/seed/  generic troubleshooting workflows (misfire, lean, catalyst, EVAP, thermostat, airflow)
fixtures/        vehicle snapshots with an `expected` block for evals
```

Not a substitute for a technician. Fuel systems hold pressure, exhausts are
hot, and the model can be wrong; the report says how confident it is and why.
