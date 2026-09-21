# autodiag: Engine Diagnostic Assistant — Design Doc

Status: Draft v0.1 (MVP scope)

## 1. Problem

A check-engine light gives a driver or technician a trouble code and little
else. Turning that into a repair today requires cross-referencing the code
with freeze-frame data, live sensor values, the vehicle's service manual, and
experience about what usually fails on that engine. The goal of autodiag is to
compress that step: read the car, retrieve the relevant service knowledge, and
return a ranked list of probable causes, the tests that best discriminate
between them, and the evidence behind each claim.

Scope is engine and emissions-related powertrain faults (P0xxx codes). Body,
chassis, and network codes are out of scope for the MVP.

## 2. Goals and non-goals

Goals (MVP):

- Read DTCs, freeze frame, readiness monitors, and engine-relevant live PIDs
  from a standard OBD-II adapter, and run identically from a recorded fixture
  when no car is present.
- Ingest service manuals, TSBs, and troubleshooting guides (PDF, Markdown,
  text) into a local searchable knowledge base with page-level citations.
- Produce a diagnosis as structured data, not prose: ranked causes with
  probabilities, next tests ordered by discriminating power, and evidence
  items each tagged with their source (OBD value, manual chunk, symptom, media).
- Accept optional audio (engine noise) and images or video frames as extra
  evidence.
- Run end to end from a CLI in under a minute, with everything except the
  LLM call working offline. When offline, still produce a rules-only report
  and let the user queue the full diagnosis for when connectivity returns.

Non-goals (MVP):

- Bidirectional controls, actuator tests, or module programming.
- Manufacturer-specific enhanced PIDs (Mode 22) and permanent DTCs (Mode 0A).
- Real-time streaming dashboards. One snapshot per diagnosis is enough.
- A mobile or web UI. The CLI and a Python API are the interface.
- Being a substitute for a technician. Output is decision support with
  explicit safety notes and confidence.

## 3. Users and primary flow

Two users: a DIY owner with a Bluetooth ELM327 dongle, and a shop technician
who wants a faster first pass. Both share one flow:

1. Plug in adapter, run `autodiag scan` to capture a snapshot (or use a
   fixture).
2. Optionally record engine audio or photos, and type free-text symptoms.
3. Run `autodiag diagnose`. The tool retrieves knowledge and calls the LLM.
4. Read the ranked causes, do the top test, feed the result back as a new
   symptom line, and rerun. Each rerun narrows the list.

## 4. Architecture

```
            ┌────────────────────┐
            │  OBD-II adapter    │  ELM327 USB / BT / WiFi
            └────────┬───────────┘
                     │ python-OBD
   ┌─────────────────▼─────────────────┐
   │ obd/   Reader (live or simulated) │──► VehicleSnapshot
   └─────────────────┬─────────────────┘
                     │
   ┌─────────────────▼─────────────────┐    ┌──────────────────────┐
   │ media/  audio features, frames    │    │ knowledge/           │
   │         (optional)                │    │  ingest → SQLite FTS │
   └─────────────────┬─────────────────┘    │  retrieve(query)     │
                     │                      └──────────┬───────────┘
   ┌─────────────────▼──────────────────────────────────▼───────────┐
   │ reasoning/  build context → Claude (structured output)         │
   │             → Diagnosis (causes, next_tests, evidence)         │
   └─────────────────────────────┬──────────────────────────────────┘
                                 │
                       cli/  rich table + JSON report
```

Four packages, one shared `models.py`. Each layer communicates only through
pydantic models so any layer can be swapped or unit-tested in isolation.

### 4.1 OBD layer (`autodiag/obd`)

`OBDReader` protocol with two implementations:

- `PythonOBDReader`: wraps python-OBD. Queries Mode 03 (stored DTCs), Mode 07
  (pending), Mode 01 live PIDs from a curated engine list (RPM, load, coolant
  and intake temp, short and long fuel trims for both banks, MAF, MAP,
  throttle, timing advance, O2 voltages, fuel pressure, module voltage,
  catalyst temp, distance since clear), Mode 02 freeze frame for the same
  PIDs, Mode 01 PID 01 readiness bits, and Mode 09 VIN. Unsupported PIDs are
  skipped, never errored.
- `SimulatedReader`: loads a JSON fixture with the same fields. Fixtures live
  in `fixtures/` and double as test cases (a P0301 misfire, a P0171 lean
  condition, more over time).

Both return `VehicleSnapshot`: vehicle info, DTC list with status, freeze
frame, live PIDs, readiness map, and a `source` string.

A small built-in table of generic SAE codes (`dtc_db.py`) fills in
descriptions when the adapter or fixture omits them.

### 4.2 Knowledge layer (`autodiag/knowledge`)

Storage: a single SQLite file (`~/.autodiag/kb.sqlite` by default) with a
`documents` table, a `chunks` table, and an FTS5 virtual table over chunk
text. FTS5 gives BM25 ranking with zero external services and no model
download, which keeps the MVP offline-capable.

Ingest (`autodiag ingest <path>...`):

- PDF via pypdf, one text block per page so page numbers survive into
  citations. Markdown and text split on headings.
- Chunks of roughly 800 tokens with 100 token overlap. Each chunk keeps
  `doc_id`, `title`, `page`, and a stable `chunk_id`.
- Re-ingesting a file with the same content hash is a no-op.

Retrieval (`retriever.py`):

- Query construction is deterministic, not LLM-driven, so it is cheap and
  testable: DTC codes and their descriptions, the vehicle make, model and
  engine, symptom keywords, and the names of any PIDs that are out of their
  normal band (for example "long term fuel trim high").
- Runs several FTS queries (one per DTC, one for symptoms, one combined),
  merges with reciprocal rank fusion, and returns the top 12 chunks with
  scores.
- Each returned chunk gets a short handle (`c1`, `c2`, ...) that the LLM must
  use when it cites manual evidence. This is what makes citations checkable.

Future, behind the same `Retriever` interface: dense embeddings for hybrid
search, and a per-vehicle filter so a Honda manual does not surface for a
Toyota.

### 4.3 Reasoning layer (`autodiag/reasoning`)

Single call to Claude with structured output. No agent loop in the MVP: one
well-built context and one schema-constrained response is faster, cheaper, and
easier to evaluate than tool use, and the discriminating tests are what turn
it into a loop with the human in it.

Model: `claude-opus-5`, adaptive thinking (default), effort `high`,
`max_tokens` 16000. Server-side refusal fallbacks enabled by default.

Context assembly, in this order for cache stability:

1. System prompt: role, the diagnostic method (evidence before conclusions,
   prefer tests that split the candidate set, cite chunk handles, call out
   contradicting evidence, never invent manual content), and the severity
   scale.
2. Retrieved chunks, each prefixed with its handle, title, and page.
3. The vehicle snapshot as compact JSON, with a derived "flags" section:
   fuel trims beyond plus or minus 10 percent, coolant below thermostat
   range, O2 stuck, MAP or MAF implausible for RPM, and so on. Flags are
   computed in code so the model reasons from them rather than rediscovering
   them.
4. Free-text symptoms.
5. Audio feature summary and any images (as image content blocks).

Output schema (`Diagnosis` in `models.py`, enforced via `output_config`
JSON schema with `additionalProperties: false`):

```
Diagnosis
  summary               2-3 plain-language sentences
  severity              drive_normally | drive_with_caution | limit_driving | do_not_drive
  causes[]              rank, title, probability (sum ~1), reasoning,
                        evidence[] {kind, detail, source}, contradicting_evidence[]
  next_tests[]          order, name, procedure, discriminates[] (cause titles),
                        expected_if_faulty, expected_if_ok, tools_needed[],
                        est_minutes, source (chunk handle if from manual)
  missing_information[] what would most raise confidence
  safety_notes[]
```

Post-processing in code: verify that every `source` handle refers to a chunk
that was actually provided (drop or flag otherwise), renormalize
probabilities, and check that every test's `discriminates` names real causes.
This is the guard against fabricated citations.

Error handling: `refusal` stop reason surfaces as a clear message, not a
crash; `max_tokens` triggers one retry with a larger budget; API errors are
caught by type (rate limit, connection, status) with the SDK's own retries.

### 4.4 Media layer (`autodiag/media`)

Audio: Claude does not accept audio, so we extract features in numpy and
pass them as text. For engine noise the useful features are duration, RMS
level, the top dominant frequencies, and the rate of repeating impulses
(ticks or knocks per second). Combined with RPM from the snapshot, the
reasoning layer can say things like "a pulse at half crank frequency
suggests a valvetrain source" versus "a pulse at crank frequency suggests
a rod or piston source". WAV is supported natively; other formats need
ffmpeg present.

Images and video: images are passed directly as vision inputs. Video is
sampled into a handful of frames with ffmpeg (when installed) and the
frames are passed as images. Typical uses: a spark plug photo, a
leaking-hose photo, a dash cluster photo.

Both inputs are optional and the pipeline is unchanged without them.

### 4.5 CLI (`autodiag/cli.py`)

Typer app with these commands:

| Command | Purpose |
|---|---|
| `scan` | Read the adapter (or `--fixture`) and save a snapshot JSON |
| `ingest PATH...` | Add manuals and guides to the knowledge base |
| `search QUERY` | Inspect what retrieval returns, for debugging the KB |
| `diagnose` | Run the full pipeline from a snapshot or fixture, print a report, write JSON |
| `diagnose --offline` | Rules-only report from DTCs, flags, and retrieved chunks; no API call (see 4.7) |
| `diagnose --queue` | Do everything except the API call and save a job for later (see 4.7) |
| `diagnose --drain` | Send queued jobs and write their reports |
| `demo` | `diagnose` against the bundled misfire fixture and seed knowledge |

Report rendering uses rich: a causes table, a next-tests table, and an
evidence list with handles resolved back to document titles and pages.

## 4.6 RAG pipeline

Retrieval-augmented generation is what connects the vehicle data to the
service knowledge. This section pulls the pieces from 4.2 and 4.3 into one
end-to-end view.

```
 OFFLINE (once per document)
 ┌──────────┐   parse    ┌──────────┐   chunk    ┌──────────┐   index   ┌──────────────┐
 │ PDF / MD │ ─────────► │ pages /  │ ─────────► │ ~800-tok │ ────────► │ SQLite       │
 │ / TXT    │  pypdf     │ sections │  overlap   │ chunks   │  FTS5     │ chunks + fts │
 └──────────┘            └──────────┘            └──────────┘           └──────────────┘

 ONLINE (per diagnosis)
 ┌──────────────┐  build    ┌──────────────┐  search  ┌──────────────┐  fuse   ┌────────────┐
 │ snapshot +   │ ────────► │ N queries    │ ───────► │ N ranked     │ ──────► │ top-12     │
 │ symptoms     │  rules    │ (DTC, sympt, │  BM25    │ result lists │  RRF    │ chunks     │
 └──────────────┘           │  vehicle...) │          └──────────────┘         │ c1 .. c12  │
                            └──────────────┘                                   └─────┬──────┘
                                                                                     │ inject
 ┌──────────────┐  validate  ┌──────────────┐  generate  ┌──────────────────────────▼──────┐
 │ Diagnosis    │ ◄───────── │ JSON with    │ ◄───────── │ prompt: system + chunks +       │
 │ (checked     │  handles   │ c# citations │  Claude    │ snapshot + flags + symptoms +   │
 │  citations)  │            └──────────────┘  Opus 5    │ media                            │
 └──────────────┘                                        └─────────────────────────────────┘
```

### Stage 1: Ingest (offline)

Input: PDF, Markdown, or plain text. Output: rows in `documents` and
`chunks`, plus an FTS5 index.

- **Parse.** PDFs are read page by page with pypdf so every chunk carries a
  page number. Markdown is split at headings so a chunk maps to a named
  section. Plain text falls back to paragraph boundaries.
- **Chunk.** Target 800 tokens with 100 token overlap, never splitting
  inside a numbered procedure step when a step boundary is available.
  Each chunk stores `doc_id`, `title`, `page` or `section`, `text`, and a
  stable `chunk_id` derived from a hash of `doc_id + position`.
- **Index.** Chunk text goes into an FTS5 virtual table with the porter
  tokenizer, so "misfiring" matches "misfire". DTC codes are indexed as
  whole tokens so `P0301` matches exactly and does not match `P0300`.
- **Dedup.** A content hash on `documents` makes re-ingest idempotent.

### Stage 2: Query construction (online, deterministic)

No LLM here. Queries are built from the snapshot by rules so retrieval is
cheap, reproducible, and unit-testable.

| Query | Built from | Example |
|---|---|---|
| One per DTC | code plus description words | `P0301 cylinder 1 misfire` |
| Symptoms | user text, stopwords removed | `rough idle cold start shake` |
| Flags | out-of-band PIDs mapped to phrases | `long term fuel trim positive lean` |
| Vehicle | make, model, engine | `Honda Accord K24` |
| Combined | DTC codes + symptom terms + vehicle | one broad query |

Flags come from a small rules table (fuel trim beyond plus or minus 10
percent, coolant below 75 degC when warm, O2 voltage stuck, MAF grams per
second implausible for RPM, and so on). The same flags are also shown to the
model, so retrieval and reasoning see the same derived facts.

### Stage 3: Search and fusion

Each query runs against FTS5 and returns its top 20 by BM25. Lists are
merged with reciprocal rank fusion (k = 60), which rewards chunks that
appear in several lists without needing score normalization across queries.
The DTC-specific queries get a weight of 2 because a chunk that names the
exact code is almost always more useful than one that only matches symptom
words. Chunks from the same page are collapsed to the best one. The top 12
survive.

Each surviving chunk is assigned a short handle `c1` through `c12` in rank
order. Handles, not titles or page numbers, are what the model cites.

### Stage 4: Context injection

The prompt is assembled in a fixed order so the stable prefix caches well:

1. System prompt (method, citation rules, severity scale). Static.
2. Retrieved chunks, formatted as:
   ```
   [c3] Misfire Isolation Workflow, section "Step 2: Swap test", p.4
   <chunk text>
   ```
3. Snapshot JSON and derived flags.
4. Symptoms text.
5. Audio feature summary and image blocks.

The citation rules in the system prompt are the heart of the augmentation:
any claim about a procedure, specification, or known failure pattern must
carry an `Evidence` item with `kind: "manual"` and `source: "c#"`. If the
model relies on its own training instead, it must label that evidence
`general_knowledge` with an empty source. This makes the boundary between
retrieved and recalled knowledge visible in the output.

### Stage 5: Generation and citation validation

Claude Opus 5 returns JSON matching the `Diagnosis` schema. Before the
result is shown:

- Every `source` that looks like a handle is checked against the set
  `{c1..c12}` provided. Unknown handles are downgraded to
  `general_knowledge` and counted; the count is reported so bad citation
  behavior is visible in evals.
- Handles are resolved back to document title and page for the report, so
  a technician can open the manual to the cited page.
- Optional strict mode: the quoted `detail` for manual evidence must appear
  as a substring (after whitespace normalization) in the referenced chunk.
  Off by default because paraphrase is legitimate, on in evals.

### What is deliberately not in the MVP pipeline

- **Dense retrieval.** No embeddings. FTS5 with porter stemming covers the
  code-heavy, part-name-heavy vocabulary of service manuals well. The
  `Retriever` interface has one method, `search(queries) -> list[Chunk]`,
  so a hybrid BM25 plus embedding retriever drops in without touching the
  reasoning layer. This is the first M2 item if evals show recall gaps on
  paraphrased symptoms.
- **Reranking.** No cross-encoder or LLM reranker. RRF plus DTC weighting is
  enough at 12 chunks.
- **Agentic retrieval.** The model cannot call `search` itself. One
  retrieve-then-generate pass keeps cost fixed and retrieval testable.
  Exposing `search` as a tool is the M2 path to "follow up on a lead"
  behavior.
- **Vehicle-scoped filtering.** All documents are searched regardless of
  make. A `vehicle` metadata filter on `documents` is planned so a Honda
  manual does not surface for a Toyota.

### Retrieval evaluation

Separate from end-to-end diagnosis evals, retrieval is measured on its own:
each fixture lists the seed chunk ids that a technician would consider
relevant, and the eval reports recall at 12 and mean reciprocal rank. This
isolates "retrieval missed it" from "the model ignored it".

## 4.7 Offline operation

Everything except the Claude call runs locally: OBD scan, ingest, retrieval,
audio and image processing. The diagnosis call is the only network
dependency, and a garage often has no signal. Two mitigations ship in M1.

### Capture now, diagnose later (queue mode)

`diagnose --queue` does every step up to the API call and writes a
self-contained job directory instead of calling the model:

```
~/.autodiag/queue/<timestamp>-<vin-or-fixture>/
  snapshot.json     vehicle snapshot
  symptoms.txt      free text
  chunks.json       the retrieved chunks with handles c1..c12
  media/            copied audio file and images
  request.json      the fully assembled prompt payload, ready to send
```

Retrieval runs at capture time so the job is frozen against the knowledge
base as it was in the garage. `diagnose --drain` later sends each queued
job, writes `diagnosis.json` and the rendered report next to it, and marks
it done. Jobs that fail (rate limit, network drop) stay in the queue and
are retried on the next drain. Nothing is ever sent silently; drain is an
explicit command.

### Rules-only fallback (offline report)

`diagnose --offline`, or an automatic fallback when the API is unreachable
and the user did not ask to queue, prints a report built entirely from local
data:

1. Each DTC with its description and status.
2. The derived flags (fuel trim out of band, coolant below thermostat range,
   and so on) with the raw PID values behind them.
3. The top retrieved manual chunks grouped by DTC, with title and page,
   so the technician can follow the standard procedure by hand.
4. Readiness monitor state, which matters for emissions testing.

This is roughly what a good code reader plus a manual index provides. It has
no ranking, probabilities, or cross-evidence reasoning, and the report header
says so explicitly so it is not mistaken for a full diagnosis. It reuses the
same snapshot, flags, and retrieval code as the online path, so it costs
almost nothing to maintain and doubles as a debugging view of what the model
would have been given.

### Connectivity detection

Before the API call, the client makes one short request with a 3 second
timeout. On failure it does not retry; it falls through to the offline report
and prints a hint about `--queue`. The SDK's own retries apply only once a
connection has been established.

### Local model backend (added to M1 as an option)

The reasoning step can run on a local open-weight model through Ollama. This
was originally deferred to M3; it was pulled forward as an opt-in because the
backend abstraction made it cheap (one HTTP call, no new dependency) and
because it gives a fully offline path for users who accept lower quality.

Design choices:

- Selection is explicit: `--backend ollama` or `AUTODIAG_BACKEND=ollama`.
  Automatic use when the Claude host is unreachable requires
  `AUTODIAG_LOCAL_FALLBACK=1`, because local output is weaker and should not
  silently replace Claude's.
- The same system prompt, user content, and JSON schema are sent. Ollama's
  `format` parameter constrains output to the schema; if the result still
  fails pydantic validation, the error is fed back once for a repair.
- Citation validation runs unchanged, so hallucinated handles are downgraded
  and counted exactly as for Claude. The report footer names the backend and
  adds a caution line for local results.
- Evaluation against the fixture set remains the gate for trusting a given
  local model. Until that exists, treat the local path as a convenience,
  not a peer of the default.

## 5. Data flow, one request

```
snapshot.json ─┐
symptoms ──────┼─► build_queries() ─► retriever.search() ─► chunks[c1..c12]
audio.wav ─────┤                                                 │
photos ────────┘                                                 ▼
                    assemble_context(snapshot, flags, chunks, symptoms, media)
                                          │
                                          ▼
                       claude-opus-5, output_config=Diagnosis schema
                                          │
                                          ▼
               validate citations, renormalize, resolve handles ─► report
```

## 6. Knowledge sources

Seeded with a small set of generic, hand-written Markdown troubleshooting
workflows so the demo works out of the box: misfire isolation (P030x),
lean conditions (P0171/P0174), catalyst efficiency (P0420), EVAP leaks
(P044x/P045x), thermostat (P0128), and MAF/MAP plausibility. These are
generic SAE-level procedures, not copied manufacturer text.

Users add their own factory service manuals, TSBs, and wiring diagrams
(as PDF) with `ingest`. Licensing of manufacturer service information is the
user's responsibility; the tool stores it locally only.

## 7. Evaluation

Correctness is judged on fixtures with a known root cause. Each fixture
carries an `expected` block (the true cause and which test confirms it).
An eval script runs `diagnose` over all fixtures and reports:

- Top-1 and top-3 hit rate on the true cause.
- Whether the first recommended test would have confirmed it.
- Citation validity rate (handles that resolve to provided chunks).

Initial fixtures are synthetic. Real captured logs replace them as they are
collected. Model and prompt changes are only accepted if these numbers do
not regress.

## 8. Security and privacy

- Snapshots include the VIN. Reports are written locally; nothing is
  uploaded except the assembled context sent to the Claude API. A
  `--no-vin` flag redacts it from the prompt.
- The knowledge base is local SQLite. No manual text leaves the machine
  except the retrieved chunks included in the prompt.
- API credentials come from the environment or `ant auth login`; nothing is
  stored in the repo.

## 9. Risks and open questions

- **Retrieval quality without embeddings.** BM25 does well on code numbers
  and part names but misses paraphrase. Mitigated by deterministic query
  expansion; embeddings are the first upgrade if evals show recall gaps.
- **Hallucinated procedures.** Mitigated by citation validation and by
  instructing the model to mark general-knowledge evidence explicitly.
- **Adapter variability.** Cheap ELM327 clones drop PIDs or time out.
  The reader skips unsupported PIDs and tolerates nulls; a `--slow` flag
  disables python-OBD fast mode.
- **Audio features are coarse.** Impulse-rate detection is a heuristic
  and noisy recordings will mislead. Audio evidence is always presented to
  the model as weak evidence.
- **Cost.** One Opus call per diagnosis, roughly 8k to 15k input tokens.
  The system prompt and seed chunks are cache-friendly, but per-vehicle
  chunks vary. Acceptable for MVP.

## 10. Milestones

1. **M1 (this MVP):** simulator, knowledge base, single-call diagnosis,
   offline rules-only report, queue and drain mode, CLI, seed knowledge,
   two fixtures, tests for OBD parsing, ingestion, retrieval, citation
   validation, offline report, and queue round-trip. Live adapter path
   written but only testable with hardware.
2. **M2:** dense hybrid retrieval, vehicle-scoped filtering, eval harness
   with 20+ fixtures, iterative sessions (feed test results back with
   conversation history).
3. **M3:** manufacturer-specific PIDs via Mode 22 tables, permanent DTCs,
   live PID capture over a drive cycle for intermittent faults, simple web UI,
   eval-based selection of a recommended local model for the Ollama backend.

## 11. Repository layout

```
auto/
  DESIGN.md
  README.md
  pyproject.toml
  autodiag/
    __init__.py
    models.py
    cli.py
    obd/        reader.py, dtc_db.py
    knowledge/  ingest.py, store.py, retriever.py
    reasoning/  prompts.py, backends.py, diagnose.py, validate.py, offline.py, queue.py
    media/      audio.py, video.py
  knowledge/seed/   *.md generic workflows
  fixtures/         *.json snapshots with expected causes
  tests/
```

## 12. Decisions log

- **Python** over TypeScript: python-OBD is the mature OBD library, and
  numpy covers audio features.
- **SQLite FTS5** over a vector database for MVP: offline, zero setup,
  strong on exact tokens like codes and part names. Interface allows a
  later hybrid.
- **Single structured call** over an agent loop: predictable cost, easy to
  evaluate, and the human performing tests is the natural loop.
- **Deterministic query building** over LLM query rewriting: testable, free,
  and DTC-driven retrieval is mostly keyword-shaped anyway.
- **Claude Opus 5** as the default model, effort `high`.
