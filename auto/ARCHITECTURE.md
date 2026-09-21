# autodiag Architecture

As-built description of the M1 code. For goals, rationale, and the roadmap
see [DESIGN.md](DESIGN.md). This document explains what each module does,
what data crosses each boundary, and where to change things.

Version: 0.1.0 (M1, with optional local backend)

## 1. System overview

autodiag turns an OBD-II snapshot plus optional symptoms and media into a
ranked differential diagnosis with cited evidence. It is a single-process
Python CLI. All state is local files. The only network dependency is one
call to the Claude API per diagnosis, or none when a local Ollama model is used.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                               autodiag CLI                               │
│  scan · ingest · kb · search · diagnose [--offline|--queue|--drain] · demo│
└───────┬──────────────────┬──────────────────┬──────────────────┬─────────┘
        │                  │                  │                  │
        ▼                  ▼                  ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐
│    obd/      │  │   knowledge/   │  │     media/     │  │  reasoning/  │
│ live adapter │  │ ingest, store, │  │ audio features │  │ flags, prompt│
│ or fixture   │  │ retriever      │  │ video frames   │  │ model call,  │
│              │  │                │  │                │  │ validate,    │
│              │  │                │  │                │  │ offline,queue│
└──────┬───────┘  └───────┬────────┘  └───────┬────────┘  └──────┬───────┘
       │                  │                   │                   │
       ▼                  ▼                   ▼                   ▼
 VehicleSnapshot    list[Chunk]         AudioFeatures       DiagnosisResult
                                        image paths
                          │
                          ▼
                 ~/.autodiag/kb.sqlite          Claude API (api.anthropic.com
                 ~/.autodiag/queue/             or ANTHROPIC_BASE_URL gateway)
                 ~/.autodiag/reports/
```

Every arrow between packages carries a pydantic model defined in
`autodiag/models.py`. No package imports another package's internals; they
share only `models.py` and `config.py`.

## 2. Package map

```
autodiag/
  __init__.py
  config.py            paths, model, tunables (env-overridable)
  models.py            all shared data types; Diagnosis is also the LLM output schema
  cli.py               Typer app, rich rendering, error-to-exit-code mapping
  obd/
    reader.py          OBDReader protocol, PythonOBDReader, SimulatedReader
    dtc_db.py          ~45 generic SAE P-codes with descriptions
  knowledge/
    store.py           SQLite schema, FTS5 index, BM25 search
    ingest.py          PDF/Markdown/text parsing and chunking
    retriever.py       deterministic query building, RRF fusion, handles
  reasoning/
    flags.py           PID rule engine producing Flag objects
    prompts.py         system prompt and user-content assembly
    backends.py        Backend protocol; ClaudeBackend (SDK), OllamaBackend (local HTTP)
    diagnose.py        prepare(), build_request(), call_model(), choose_backend()
    validate.py        citation and probability post-checks
    offline.py         rules-only text report
    queue.py           job directories, enqueue, drain
  media/
    audio.py           WAV load, FFT peaks, impulse-rate autocorrelation
    video.py           ffmpeg frame sampling
knowledge/seed/        six generic troubleshooting workflows (Markdown)
fixtures/              vehicle snapshots with an `expected` block
tests/                 39 tests, no network
```

## 3. Data model

All types live in `models.py` and extend `StrictModel`, which sets
`extra="forbid"`. This matters for the output types: pydantic emits
`additionalProperties: false` for them, which the Claude structured-output
feature requires.

### 3.1 Input side

```
VehicleSnapshot
  vehicle: VehicleInfo        vin, year, make, model, engine, mileage
  dtcs: list[DTC]             code, description, status ∈ {stored, pending, permanent}
  freeze_frame: list[PIDValue] name, value (float|str|None), unit
  live: list[PIDValue]
  readiness: dict[str, str]   MIL, DTC_count, <monitor>: complete|incomplete
  source: str                 "obd:<port>" or "simulated:<file>"

Flag                          derived observation, computed in code
  name, severity ∈ {info, warn}, detail, pids, search_terms

AudioFeatures
  path, duration_s, rms_db, dominant_hz[], pulse_rate_hz?, notes[]

Chunk                         one retrieved passage
  chunk_id, doc_id, title, page?, section?, text, score, handle ("c1".."c12")
```

### 3.2 Output side (the LLM contract)

```
Diagnosis
  summary: str
  severity ∈ {drive_normally, drive_with_caution, limit_driving, do_not_drive}
  causes: list[Cause]
    rank, title, probability, reasoning,
    evidence: list[Evidence]        kind ∈ {obd, manual, symptom, audio, image, general_knowledge}
                                    detail, source (handle | PID/DTC name | "")
    contradicting_evidence: list[str]
  next_tests: list[NextTest]
    order, name, procedure, discriminates: list[cause title],
    expected_if_faulty, expected_if_ok, tools_needed, est_minutes, source (handle | "")
  missing_information: list[str]
  safety_notes: list[str]

DiagnosisResult               what the CLI renders and saves
  diagnosis, chunks, flags, validation: ValidationReport, model, backend, usage, fallback_model?
```

`Diagnosis.model_json_schema()` is passed verbatim as the
`output_config.format.schema`, so the pydantic class is the single source of
truth for what the model must return.

## 4. Module details

### 4.1 obd/

`OBDReader` is a `Protocol` with `read() -> VehicleSnapshot` and `close()`.

**PythonOBDReader** wraps python-OBD. On construction it opens the adapter
(auto-detect or explicit port) and raises `ConnectionError` if not connected.
`read()` issues, in order: `GET_DTC` (Mode 03), `GET_CURRENT_DTC` (Mode 07),
the 25 PIDs in `ENGINE_PIDS` (Mode 01), the same PIDs with a `DTC_` prefix
(Mode 02 freeze frame), `STATUS` (Mode 01 PID 01), and `VIN` (Mode 09). Every
query goes through `_query()`, which returns `None` when the command is
unsupported or the response is null, so a cheap adapter that drops PIDs
degrades to a smaller snapshot rather than an error. Pint quantities are
reduced to `magnitude` and `units`.

**SimulatedReader** loads a fixture JSON with the same five sections. Codes
without a description are filled from `dtc_db.describe()`. The optional
`expected` block is ignored here and reserved for the eval harness.

`open_reader(port, fixture, fast)` picks the implementation.

### 4.2 knowledge/

**store.py.** One SQLite file, three tables:

```
documents(doc_id PK, title, path, content_hash UNIQUE, ingested_at)
chunks(chunk_id PK, doc_id FK, position, page, section, text)
chunks_fts  FTS5 external-content table over chunks.text
            tokenize='porter unicode61'
```

Insert and delete triggers keep the FTS index in sync. `search(query, limit)`
runs `MATCH` ordered by `bm25()` and flips the sign so higher is better. A
malformed FTS expression returns an empty list instead of raising.

**ingest.py.** `parse_file()` dispatches on suffix: PDF via pypdf (one
`Section` per page), Markdown split at headings (first H1 becomes the title),
text as one section. `chunk_text()` targets 800 approximate tokens with 100
overlap; it splits on blank lines and on numbered or bulleted step
boundaries so a procedure step is never cut in half, and carries trailing
units forward as overlap. `ingest_path()` walks files or directories, skips a
file whose SHA-256 is already stored unless `force=True`, and returns
`(filename, n_chunks, status)` per file.

**retriever.py.** `build_queries()` is pure and deterministic:

| Query kind | Source | Weight |
|---|---|---|
| dtc (one per code) | `"p0301" OR desc words` | 2.0 |
| symptoms | user text minus stopwords | 1.0 |
| flag (one per flag with search_terms) | flag.search_terms | 1.0 |
| vehicle | make, model, engine | 0.5 |
| combined | codes + symptom terms + vehicle | 1.0 |

`Retriever.search()` runs each query for the top 20, fuses with weighted
reciprocal rank fusion (`weight / (60 + rank)`), collapses to one chunk per
(document, page), truncates to 12, and assigns handles `c1..c12` in rank
order. `search_text()` is the single-query path used by the `search`
command.

### 4.3 reasoning/

**flags.py.** `compute_flags(snapshot)` runs a fixed set of rules over both
the live and freeze-frame sections and returns `Flag` objects. Rules:

- Fuel trim: LTFT beyond ±10% or LTFT+STFT beyond ±15% flags lean or rich
  per bank; a bank difference over 10% flags a bank-specific cause.
- Thermal: coolant under 75 C after 300 s run time, coolant over 110 C,
  implausible intake temperature.
- Airflow at idle (500 to 1200 rpm, stationary): MAF under 1.5 or over 8 g/s;
  MAP over 45 kPa flags weak vacuum, otherwise an `info` flag records normal
  vacuum because that absence is evidence too.
- O2: upstream near a rail, downstream steady high.
- Charging voltage outside 13.0 to 15.0 V with engine running.
- DTC patterns: single vs multiple cylinder misfire codes, lean plus misfire.
- Readiness: monitors incomplete with under 100 km since clear.

Each flag carries `search_terms`, which become a retrieval query, and
`detail`, which goes into the prompt. Retrieval and reasoning therefore see
the same derived facts.

**prompts.py.** `SYSTEM_PROMPT` is a constant (never interpolated) so it
caches. `build_user_content()` returns content blocks in fixed order:
retrieved chunks with handles, snapshot JSON (sorted keys), flags, symptoms,
audio summary, then image blocks, then a one-line instruction. Audio is
rendered as text and related to crank frequency from live RPM. Images are
base64 blocks; unsupported MIME types are skipped.

**backends.py.** A `Backend` has a `name`, `is_available()` (3 s TCP probe
to its host), and `complete(system, user_blocks, schema, max_tokens) ->
BackendResponse` where the response carries `text`, `model`, `usage`,
`fallback_model`, and `truncated`.

- `ClaudeBackend` wraps the Anthropic SDK. It builds the request shown in
  section 5, maps `stop_reason == refusal` to `BackendRefused`, and reads
  fallback blocks and cache usage.
- `OllamaBackend` posts to `{OLLAMA_HOST}/api/chat` with the standard
  library only. System and user text become two chat messages, image
  blocks become the message's `images` list, and the JSON schema goes in
  Ollama's `format` field for constrained decoding. `done_reason ==
  "length"` maps to `truncated`; a 404 with "not found" becomes a "run
  `ollama pull`" message; connection errors become `BackendUnavailable`.
- `make_backend(name)` selects by name or `AUTODIAG_BACKEND`.

**diagnose.py.** The pipeline is split so the offline paths reuse it:

```
prepare(snapshot, store, symptoms, audio, images, redact_vin)
   -> compute_flags -> build_queries -> Retriever.search -> Prepared

build_request(Prepared, backend) -> the backend's request dict (for inspection)

is_online(backend) -> bool          delegates to backend.is_available()

call_model(Prepared, strict, backend) -> DiagnosisResult
   backend.complete -> BackendRefused -> DiagnosisRefused
                    -> BackendUnavailable -> Offline
   truncated -> one retry at 4x max_tokens
   json.loads + Diagnosis.model_validate
       on failure with a non-Claude backend: one repair round-trip that
       appends the validation error and the bad output, then re-parse
       (Claude's output_config guarantees valid JSON, so it is not retried)
   -> validate_diagnosis -> DiagnosisResult with backend name

choose_backend(requested) -> (Backend, note)
   explicit --backend, else AUTODIAG_BACKEND; if that is Claude and
   unreachable and AUTODIAG_LOCAL_FALLBACK=1 and Ollama answers, use Ollama
   and return a note for the CLI to print

diagnose(...) = prepare + is_online + call_model
```

`Prepared` serializes to and from JSON (`to_json`, `from_json`) so a queued
job can be rebuilt later. `call_model` always rebuilds the request from
`Prepared` and the current backend, so a change of model, gateway, or
backend between enqueue and drain takes effect.

**validate.py.** `validate_diagnosis()` mutates the `Diagnosis` in place:

1. Every `manual` evidence item whose `source` is not a provided handle is
   downgraded to `general_knowledge` and counted. In strict mode the first
   60 characters of `detail`, or any quoted fragment, must also appear in
   the chunk text.
2. Test `source` handles that do not resolve are blanked and counted.
3. `discriminates` entries that match no cause title are recorded.
4. Probabilities are renormalized if they sum outside 0.98 to 1.02.
5. Causes are re-ranked by probability; tests re-numbered by order.

The returned `ValidationReport` is stored in `DiagnosisResult` and printed
at the bottom of every report.

**offline.py.** `render_offline_report()` produces plain text: header
stating it is rules-only, vehicle line, codes, flags with raw PID values
(freeze-frame values first for freeze-frame flags), readiness, then chunks
grouped by which DTC appears in their text. No API call.

**queue.py.** A job is a directory `~/.autodiag/queue/<stamp>-<id>/`:

```
prepared.json   Prepared.to_json(); the frozen retrieval and flags
request.json    inspection copy of the request with messages elided
media/          copied audio and image files; paths in prepared.json are rewritten
diagnosis.json  DiagnosisResult, written on success
report.txt      rendered report, written on success
error.txt       last failure; deleted on success
```

`drain()` iterates pending jobs, calls `call_model` on the chosen backend,
and catches `DiagnosisRefused`, `Offline`, `RuntimeError`,
`APIConnectionError`, `RateLimitError`, and `APIStatusError` per job so one
failure does not stop the rest. Anything
uncaught (for example missing credentials) propagates to the CLI.

### 4.4 media/

**audio.py.** WAV is read with the standard library; other formats are
converted with ffmpeg if present. Features: duration, RMS in dBFS, up to
four dominant spectral peaks between 20 and 4000 Hz (Hann window, local
maxima at least 15 Hz apart), and `pulse_rate_hz` from the autocorrelation
of a rectified, smoothed, 1 kHz envelope. The impulse detector takes the
shortest local maximum within 80% of the strongest peak in the 2 to 80 Hz
range, which avoids reporting the subharmonic. It returns `None` when
periodicity is under 0.25. Notes warn about short, quiet, or clipped input.

**video.py.** `extract_frames()` samples `n` frames at evenly spaced
timestamps via ffprobe and ffmpeg, falling back to one frame every two
seconds when duration is unknown. Output paths feed the image list.

### 4.5 cli.py

Typer commands map to the pipeline as follows:

| Command | Calls |
|---|---|
| `scan` | `open_reader().read()` then write snapshot JSON |
| `ingest` | `ingest_path()` per argument |
| `kb` | `store.list_documents()` |
| `search` | `Retriever.search_text()` |
| `diagnose` | `prepare()`, then one of: `render_offline_report`, `q.enqueue`, `call_model` |
| `diagnose --drain` | `q.drain(render=render_result)` |
| `queue-list` | `q.list_jobs()` |
| `demo` | `ingest_path(seed)` then `diagnose(fixture=p0301)` |

`diagnose` decision order: `--drain` first; else load snapshot (file,
fixture, or adapter); analyze audio and video; `prepare`; then `--queue`,
else `--offline` gives the offline report; else `choose_backend(--backend)`,
and if that backend's probe fails the offline report, else the model call.
A local-model result prints a caution line after the footer. Rendering uses rich tables with all model text passed through
`escape()` so bracketed tokens are not read as markup.

Exit codes: 0 success; 1 offline fallback after a mid-request connection
drop or a drain with no connectivity; 2 refusal, credential, rate-limit, or
API status errors.

## 5. Request to the model

Exactly one request per diagnosis (two if the first hits `max_tokens`, and
for the Ollama backend a third if the JSON needs one repair).

### 5.1 Claude backend (default)

```
POST {ANTHROPIC_BASE_URL}/v1/messages           via client.beta.messages.create
model:          AUTODIAG_MODEL              default claude-opus-5
max_tokens:     AUTODIAG_MAX_TOKENS         default 16000
system:         [SYSTEM_PROMPT, cache_control ephemeral]
messages:       one user turn: text block (chunks, snapshot, flags, symptoms, audio),
                optional image blocks, closing instruction
output_config:  effort AUTODIAG_EFFORT (high), format json_schema = Diagnosis schema
betas:          ["server-side-fallback-2026-07-01"]   only if AUTODIAG_FALLBACKS != 0
fallbacks:      "default"                              only if AUTODIAG_FALLBACKS != 0
thinking:       omitted (adaptive by default on this model)
```

Typical size for the demo: 8k to 12k input tokens, 2k to 4k output tokens.
The system prompt is cache-stable; the user turn varies per vehicle.

### 5.2 Ollama backend (local)

```
POST {OLLAMA_HOST}/api/chat                       standard library urllib
model:     AUTODIAG_LOCAL_MODEL          default qwen2.5:14b
messages:  [{role: system, content: SYSTEM_PROMPT},
            {role: user, content: <all text blocks joined>, images: [<base64>...]}]
format:    Diagnosis JSON schema         Ollama constrained decoding
stream:    false
options:   num_predict AUTODIAG_MAX_TOKENS, num_ctx AUTODIAG_LOCAL_NUM_CTX (16384),
           temperature 0.2
```

No prompt caching, no refusal fallbacks, no effort control. Response fields
used: `message.content`, `model`, `done_reason`, `prompt_eval_count`,
`eval_count`. Code fences around the JSON are stripped before parsing. The
context window must hold the whole prompt; at the default 16k tokens the
demo fits with room, but many large PDFs in retrieval could exceed it, in
which case Ollama silently truncates the prompt from the front. Raise
`AUTODIAG_LOCAL_NUM_CTX` if the model's context allows.

## 6. Storage and files

| Path | Contents | Written by |
|---|---|---|
| `~/.autodiag/kb.sqlite` | documents, chunks, FTS index | `ingest`, `demo` |
| `~/.autodiag/queue/<job>/` | see 4.3 queue layout | `diagnose --queue`, `--drain` |
| `~/.autodiag/reports/diagnosis-<stamp>.json` and `.txt` | `DiagnosisResult` and plain report | `diagnose` |
| `./snapshot.json` (default) | `VehicleSnapshot` | `scan` |

`AUTODIAG_HOME` relocates the first three. Nothing else is written.

## 7. Configuration

All in `config.py`, all overridable by environment variable:

| Variable | Default | Effect |
|---|---|---|
| `AUTODIAG_HOME` | `~/.autodiag` | root for kb, queue, reports |
| `AUTODIAG_KB`, `AUTODIAG_QUEUE`, `AUTODIAG_REPORTS` | under HOME | individual overrides |
| `AUTODIAG_MODEL` | `claude-opus-5` | model ID sent in the request |
| `AUTODIAG_EFFORT` | `high` | `output_config.effort` |
| `AUTODIAG_MAX_TOKENS` | `16000` | output cap |
| `AUTODIAG_FALLBACKS` | `1` | `0` drops the fallback beta and parameter |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | SDK endpoint and probe host |
| `ANTHROPIC_API_KEY` | unset | SDK credential (or an `ant auth login` profile) |
| `AUTODIAG_BACKEND` | `claude` | `claude` or `ollama` |
| `AUTODIAG_LOCAL_FALLBACK` | `0` | `1` uses Ollama automatically when the Claude host is unreachable |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `AUTODIAG_LOCAL_MODEL` | `qwen2.5:14b` | Ollama model name |
| `AUTODIAG_LOCAL_NUM_CTX` | `16384` | local context window in tokens |
| `AUTODIAG_LOCAL_TIMEOUT` | `600` | seconds to wait for a local response |

Retrieval constants (chunk size 800/100, per-query limit 20, top-k 12,
RRF k 60, DTC weight 2.0) and the 3 s probe timeout are module constants
in the same file.

## 8. Error handling

| Condition | Where caught | Behavior |
|---|---|---|
| Adapter not connected | `PythonOBDReader.__init__` | `ConnectionError` with guidance |
| Unsupported PID or null response | `_query` | skipped |
| Bad FTS expression | `store.search` | empty result |
| API host unreachable before call | `is_online` | offline report, hint to `--queue` |
| No credentials | `cli` (SDK raises `TypeError` at request time) | message, exit 2 |
| `stop_reason == refusal` | `call_model` | `DiagnosisRefused`, exit 2 |
| `stop_reason == max_tokens` | `call_model` | one retry at 4x, then error |
| Invalid JSON or schema mismatch, Claude | `call_model` | `ValidationError` propagates (loud failure), exit 2 |
| Invalid JSON or schema mismatch, Ollama | `call_model` | one repair round-trip, then propagates, exit 2 with a hint to try a larger model |
| Ollama model not pulled | `OllamaBackend._post` | `RuntimeError` naming the `ollama pull` command, exit 2 |
| Ollama unreachable mid-request | `OllamaBackend._post` | `Offline`, offline report, exit 1 |
| Bad citation handles | `validate_diagnosis` | downgraded, counted, shown |
| Per-job API failure in drain | `queue.drain` | `error.txt`, job retained |

The SDK's own retries (2, exponential backoff) apply to 429 and 5xx once a
connection exists.

## 9. Testing

39 tests under `tests/`, all offline, about 2 s:

- `test_obd.py`: fixture parsing, description fill-in, VIN redaction.
- `test_knowledge.py`: idempotent ingest, heading preservation, chunk
  boundaries and overlap, query shapes and weights, retrieval ranking for
  both fixtures, malformed queries, directory ingest.
- `test_reasoning.py`: citation downgrade and renormalization, strict mode,
  flag rules on both fixtures, schema strictness, request shape in normal
  and gateway modes, `Prepared` round-trip, offline report content, image
  block filtering, probe host parsing.
- `test_queue.py`: `call_model` against a fake client (parse, validate,
  usage, refusal), enqueue with media copy, drain with failure then success.
- `test_backends.py`: Ollama request shape including images and schema,
  fence stripping and the JSON repair round-trip, truncation retry, the
  "model not pulled" and unreachable paths, Claude not retrying on bad JSON,
  backend selection, and local-fallback choice.
- `test_audio.py`: tone detection, impulse rate, noise rejection, end to end
  WAV analysis.

The model call is tested only through a fake client. Live behavior
(citation discipline, `discriminates` matching, structured output through a
gateway) is verified manually until the M2 eval harness exists.

## 10. Extension points

- **New reader**: implement `read()` and `close()`; return a
  `VehicleSnapshot`. Nothing downstream changes.
- **New flag rule**: add a function in `flags.py` returning `Flag` objects
  and call it from `compute_flags`. Give it `search_terms` so retrieval
  benefits too.
- **Dense or hybrid retrieval**: subclass or replace `Retriever`; keep
  `search(queries) -> list[Chunk]` with handles assigned. Add an embedding
  column to `chunks` and a second scorer inside the fusion loop.
- **Vehicle-scoped filtering**: add `make`/`model` columns to `documents`
  and a `WHERE` clause in `store.search`.
- **New evidence type**: add to `EvidenceKind` in `models.py`; the schema
  updates automatically.
- **Agentic retrieval**: expose `Retriever.search_text` as a tool and run
  the tool runner instead of one `create` call. `Prepared` and
  `validate_diagnosis` still apply.
- **Different model or provider**: `AUTODIAG_MODEL` and `ANTHROPIC_BASE_URL`
  for anything that speaks the Anthropic Messages API; `AUTODIAG_BACKEND=ollama`
  for any model Ollama serves. Another provider is a third class in
  `backends.py` implementing `is_available` and `complete`, plus one line
  in `make_backend`.

## 11. Known limitations (M1)

- Retrieval is sparse only; paraphrased symptoms may miss relevant chunks.
- No Mode 0A permanent codes, no Mode 22 manufacturer PIDs.
- One snapshot per diagnosis; no drive-cycle capture for intermittent faults.
- Audio features are coarse and presented to the model as weak evidence.
- Video needs ffmpeg on PATH; non-WAV audio does too.
- The fixture simulator works at the decoded PID level, so python-OBD's
  frame parsing is only exercised with real hardware.
- Structured output depends on `output_config` reaching the API unchanged;
  a gateway that strips it causes a loud parse failure, not a fallback.
- The local backend has not been evaluated against the fixtures; which
  Ollama model is good enough is an open question until the M2 eval
  harness exists. Small models tend to cite handles loosely, which shows up
  as a higher `invalid_citations` count.
