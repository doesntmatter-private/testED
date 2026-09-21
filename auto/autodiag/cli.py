"""autodiag command line."""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import pydantic
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import config
from .knowledge import KnowledgeStore, Retriever, ingest_path
from .models import DiagnosisResult, VehicleSnapshot
from .obd import open_reader
from .reasoning import diagnose as _diagnose_mod
from .reasoning.diagnose import DiagnosisRefused, Offline, call_model, choose_backend, prepare
from .reasoning.offline import render_offline_report
from .reasoning import queue as q

app = typer.Typer(help="OBD-II + service knowledge + LLM engine diagnosis.", no_args_is_help=True)
console = Console()


def _store() -> KnowledgeStore:
    config.ensure_dirs()
    return KnowledgeStore(config.KB_PATH)


NO_CREDS_MSG = (
    "[red]No API credentials found.[/] Run `ant auth login`, or set ANTHROPIC_API_KEY. "
    "Use --offline for a rules-only report or --queue to save this job for later."
)


def _is_missing_credentials(e: Exception) -> bool:
    # The SDK raises a bare TypeError at request time when no credential source resolves.
    return isinstance(e, TypeError) and "authentication method" in str(e)


def _load_snapshot(snapshot: Path | None, fixture: Path | None, port: str | None, slow: bool) -> VehicleSnapshot:
    if snapshot:
        return VehicleSnapshot.model_validate_json(snapshot.read_text())
    reader = open_reader(port=port, fixture=fixture, fast=not slow)
    try:
        return reader.read()
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_result(result: DiagnosisResult) -> str:
    """Plain-text report (used for files and the queue)."""
    d = result.diagnosis
    by_handle = {c.handle: c for c in result.chunks}
    lines = [f"Severity: {d.severity}", "", d.summary, "", "== Ranked causes =="]
    for c in d.causes:
        lines.append(f"{c.rank}. {c.title}  ({c.probability:.0%})")
        lines.append(f"   {c.reasoning}")
        for e in c.evidence:
            src = e.source
            if e.kind == "manual" and src in by_handle:
                ch = by_handle[src]
                src = f"{src} = {ch.title}" + (f", {ch.location()}" if ch.location() else "")
            lines.append(f"   - [{e.kind}{': ' + src if src else ''}] {e.detail}")
        for ce in c.contradicting_evidence:
            lines.append(f"   x {ce}")
    lines.append("")
    lines.append("== Next tests ==")
    for t in d.next_tests:
        lines.append(f"{t.order}. {t.name}  (~{t.est_minutes} min; {', '.join(t.tools_needed) or 'no special tools'})")
        lines.append(f"   {t.procedure}")
        lines.append(f"   Discriminates: {', '.join(t.discriminates)}")
        lines.append(f"   If faulty: {t.expected_if_faulty}")
        lines.append(f"   If OK: {t.expected_if_ok}")
        if t.source and t.source in by_handle:
            ch = by_handle[t.source]
            lines.append(f"   Source: {t.source} = {ch.title}" + (f", {ch.location()}" if ch.location() else ""))
    if d.missing_information:
        lines.append("")
        lines.append("== Missing information ==")
        lines.extend(f"- {m}" for m in d.missing_information)
    if d.safety_notes:
        lines.append("")
        lines.append("== Safety ==")
        lines.extend(f"- {s}" for s in d.safety_notes)
    lines.append("")
    v = result.validation
    lines.append(
        f"backend={result.backend}  model={result.model}"
        + (f" (fallback from {config.MODEL})" if result.fallback_model else "")
        + f"  tokens in/out={result.usage.get('input_tokens')}/{result.usage.get('output_tokens')}"
        + f"  cache_read={result.usage.get('cache_read_input_tokens', 0)}"
        + f"  invalid_citations={v.invalid_citations}"
    )
    if result.backend != "claude":
        lines.append("Produced by a local model; treat probabilities and citations with extra care.")
    return "\n".join(lines)


def print_result(result: DiagnosisResult) -> None:
    d = result.diagnosis
    by_handle = {c.handle: c for c in result.chunks}
    sev_color = {
        "drive_normally": "green",
        "drive_with_caution": "yellow",
        "limit_driving": "dark_orange",
        "do_not_drive": "red",
    }[d.severity]
    console.print(Panel(escape(d.summary), title=f"[{sev_color}]{d.severity.replace('_', ' ')}[/]", expand=False))

    t = Table(title="Ranked causes", show_lines=True)
    t.add_column("#", width=2)
    t.add_column("Cause", style="bold")
    t.add_column("P", width=5, justify="right")
    t.add_column("Reasoning and evidence")
    for c in d.causes:
        ev_lines = []
        for e in c.evidence:
            src = e.source
            if e.kind == "manual" and src in by_handle:
                ch = by_handle[src]
                src = f"{src} {ch.title}" + (f" {ch.location()}" if ch.location() else "")
            ev_lines.append(f"[dim]{escape(e.kind + (' ' + src if src else ''))}:[/] {escape(e.detail)}")
        for ce in c.contradicting_evidence:
            ev_lines.append(f"[red]against:[/] {escape(ce)}")
        t.add_row(str(c.rank), escape(c.title), f"{c.probability:.0%}", escape(c.reasoning) + "\n" + "\n".join(ev_lines))
    console.print(t)

    t2 = Table(title="Next tests (in order)", show_lines=True)
    t2.add_column("#", width=2)
    t2.add_column("Test", style="bold")
    t2.add_column("Procedure")
    t2.add_column("Discriminates")
    t2.add_column("Min", width=4, justify="right")
    for nt in d.next_tests:
        proc = escape(nt.procedure) + f"\n[green]faulty:[/] {escape(nt.expected_if_faulty)}\n[green]ok:[/] {escape(nt.expected_if_ok)}"
        if nt.source and nt.source in by_handle:
            ch = by_handle[nt.source]
            proc += f"\n[dim]{escape(f'source {nt.source}: {ch.title} {ch.location()}')}[/]"
        t2.add_row(str(nt.order), escape(nt.name), proc, escape("\n".join(nt.discriminates)), str(nt.est_minutes))
    console.print(t2)

    if d.missing_information:
        console.print("[bold]Missing information:[/] " + escape("; ".join(d.missing_information)))
    if d.safety_notes:
        console.print("[bold red]Safety:[/] " + escape("; ".join(d.safety_notes)))
    v = result.validation
    console.print(
        f"[dim]backend={result.backend} model={result.model}"
        + (f" (fallback from {config.MODEL})" if result.fallback_model else "")
        + f" tokens in/out={result.usage.get('input_tokens')}/{result.usage.get('output_tokens')}"
        + f" cache_read={result.usage.get('cache_read_input_tokens', 0)} invalid_citations={v.invalid_citations}[/]"
    )
    if result.backend != "claude":
        console.print("[yellow]Produced by a local model; treat probabilities and citations with extra care.[/]")
    if v.downgraded:
        console.print("[yellow]Downgraded citations:[/] " + escape("; ".join(v.downgraded)))


def _save_report(result: DiagnosisResult, out: Path | None) -> Path:
    config.ensure_dirs()
    if out is None:
        out = config.REPORTS_DIR / f"diagnosis-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=1))
    out.with_suffix(".txt").write_text(render_result(result))
    return out


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def scan(
    port: str | None = typer.Option(None, help="Serial port, e.g. /dev/tty.OBDII or COM3. Auto-detect if omitted."),
    fixture: Path | None = typer.Option(None, exists=True, help="Replay a JSON fixture instead of a live adapter."),
    out: Path = typer.Option(Path("snapshot.json"), help="Where to write the snapshot."),
    slow: bool = typer.Option(False, help="Disable python-OBD fast mode for flaky adapters."),
):
    """Read DTCs, freeze frame, readiness and live engine PIDs; save a snapshot."""
    snap = _load_snapshot(None, fixture, port, slow)
    out.write_text(snap.model_dump_json(indent=1))
    console.print(f"[green]Snapshot saved[/] to {out}  ({snap.source})")
    t = Table(title="Trouble codes")
    t.add_column("Code")
    t.add_column("Status")
    t.add_column("Description")
    for d in snap.dtcs:
        t.add_row(d.code, d.status, d.description)
    console.print(t if snap.dtcs else "[dim]no DTCs stored[/]")
    console.print(f"[dim]{len(snap.live)} live PIDs, {len(snap.freeze_frame)} freeze-frame PIDs, readiness: {snap.readiness or 'n/a'}[/]")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, help="PDF, Markdown, or text files or directories."),
    force: bool = typer.Option(False, help="Re-ingest even if unchanged."),
):
    """Add service manuals, TSBs and troubleshooting guides to the knowledge base."""
    store = _store()
    t = Table(title=f"Ingest into {config.KB_PATH}")
    t.add_column("File")
    t.add_column("Chunks", justify="right")
    t.add_column("Status")
    for p in paths:
        for name, n, status in ingest_path(store, p, force=force):
            t.add_row(name, str(n), status)
    console.print(t)
    console.print(f"[dim]{store.count_chunks()} chunks total[/]")


@app.command()
def kb():
    """List documents in the knowledge base."""
    store = _store()
    t = Table(title=str(config.KB_PATH))
    t.add_column("Title")
    t.add_column("Chunks", justify="right")
    t.add_column("Path", style="dim")
    for row in store.list_documents():
        t.add_row(row["title"], str(row["n_chunks"]), row["path"])
    console.print(t)


@app.command()
def search(query: str, k: int = typer.Option(8, help="Results to show.")):
    """Search the knowledge base directly (debugging retrieval)."""
    store = _store()
    for c in Retriever(store, top_k=k).search_text(query, limit=k):
        loc = c.location()
        title = escape(f"[{c.handle}] {c.title}" + (f" ({loc})" if loc else ""))
        console.print(Panel(escape(c.text.strip()[:700]), title=title, subtitle=f"bm25 {c.score:.2f}"))


@app.command()
def diagnose(
    snapshot: Path | None = typer.Option(None, exists=True, help="Snapshot JSON from `scan`."),
    fixture: Path | None = typer.Option(None, exists=True, help="Fixture JSON to simulate a vehicle."),
    port: str | None = typer.Option(None, help="Read a live adapter instead."),
    symptoms: str = typer.Option("", help="Owner-reported symptoms, free text."),
    audio: Path | None = typer.Option(None, exists=True, help="Engine audio recording (WAV, or anything with ffmpeg)."),
    image: list[Path] = typer.Option([], help="Photo(s) to include. Repeatable."),
    video: Path | None = typer.Option(None, exists=True, help="Video to sample frames from (needs ffmpeg)."),
    offline: bool = typer.Option(False, help="Rules-only report; never call the API."),
    queue: bool = typer.Option(False, help="Prepare everything and save a job for later instead of calling the API."),
    drain: bool = typer.Option(False, help="Send all queued jobs now."),
    no_vin: bool = typer.Option(False, help="Redact the VIN from what is sent to the API."),
    strict: bool = typer.Option(False, help="Strict citation check: quoted manual evidence must appear in the chunk."),
    out: Path | None = typer.Option(None, help="Write the JSON report here (default ~/.autodiag/reports/)."),
    slow: bool = typer.Option(False, help="Disable python-OBD fast mode."),
    backend: str | None = typer.Option(
        None, help="Model backend: 'claude' (default) or 'ollama' for a local model. Overrides AUTODIAG_BACKEND."
    ),
    local_model: str | None = typer.Option(None, help="Ollama model name, e.g. qwen2.5:14b. Overrides AUTODIAG_LOCAL_MODEL."),
):
    """Retrieve knowledge and produce ranked causes, next tests and evidence."""
    store = _store()
    if local_model:
        config.LOCAL_MODEL = local_model

    if drain:
        _drain(store, backend)
        return

    if not (snapshot or fixture or port is not None):
        # Default to a live adapter with auto-detect if nothing else is given.
        port = port or ""
    snap = _load_snapshot(snapshot, fixture, port or None, slow)

    audio_feats = None
    if audio:
        from .media import analyze_audio

        audio_feats = analyze_audio(audio)
        console.print(f"[dim]audio: {audio_feats.duration_s}s, pulse {audio_feats.pulse_rate_hz or 'n/a'} Hz, tones {audio_feats.dominant_hz}[/]")
    image_paths = [str(p) for p in image]
    if video:
        from .media import extract_frames

        frames = extract_frames(video)
        console.print(f"[dim]video: {len(frames)} frames extracted[/]")
        image_paths.extend(frames)

    prepared = prepare(snap, store, symptoms, audio_feats, image_paths, redact_vin=no_vin)
    console.print(f"[dim]{len(prepared.flags)} flags, {len(prepared.chunks)} chunks retrieved from {store.count_chunks()} indexed[/]")

    if queue:
        job = q.enqueue(prepared)
        console.print(f"[green]Queued[/] {job.path}\nRun `autodiag diagnose --drain` when online.")
        return

    if offline:
        console.print(render_offline_report(snap, prepared.flags, prepared.chunks), markup=False, highlight=False)
        return

    chosen, note = choose_backend(backend)
    if note:
        console.print(f"[yellow]{note}[/]")
    if not chosen.is_available():
        where = "API host" if chosen.name == "claude" else f"Ollama at {config.OLLAMA_HOST}"
        console.print(f"[yellow]{where} unreachable; falling back to the offline report. Use --queue to save this job for later.[/]")
        console.print(render_offline_report(snap, prepared.flags, prepared.chunks), markup=False, highlight=False)
        return
    if chosen.name == "ollama":
        console.print(f"[dim]local model {chosen.model} via {chosen.host}; expect slower, less reliable citations[/]")

    try:
        result = call_model(prepared, strict=strict, backend=chosen)
    except Offline as e:
        console.print(f"[yellow]{e}. Falling back to the offline report.[/]")
        console.print(render_offline_report(snap, prepared.flags, prepared.chunks), markup=False, highlight=False)
        raise typer.Exit(1)
    except (json.JSONDecodeError, pydantic.ValidationError) as e:
        console.print(f"[red]The model returned output that does not match the diagnosis schema:[/] {escape(str(e)[:400])}")
        if chosen.name == "ollama":
            console.print("Try a larger local model (--local-model), or the Claude backend.")
        raise typer.Exit(2)
    except RuntimeError as e:
        console.print(f"[red]{escape(str(e))}[/]")
        raise typer.Exit(2)
    except DiagnosisRefused as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2)
    except anthropic.AuthenticationError:
        console.print("[red]API credentials were rejected. Check ANTHROPIC_API_KEY or run `ant auth login`.[/]")
        raise typer.Exit(2)
    except TypeError as e:
        if not _is_missing_credentials(e):
            raise
        console.print(NO_CREDS_MSG)
        raise typer.Exit(2)
    except anthropic.RateLimitError:
        console.print("[red]Rate limited. Try again shortly or use --queue.[/]")
        raise typer.Exit(2)
    except anthropic.APIConnectionError:
        console.print("[yellow]Connection dropped mid-request. Falling back to the offline report.[/]")
        console.print(render_offline_report(snap, prepared.flags, prepared.chunks), markup=False, highlight=False)
        raise typer.Exit(1)
    except anthropic.APIStatusError as e:
        console.print(f"[red]API error {e.status_code}: {e.message}[/]")
        raise typer.Exit(2)

    print_result(result)
    path = _save_report(result, out)
    console.print(f"[dim]saved {path} and {path.with_suffix('.txt')}[/]")


def _drain(store: KnowledgeStore, backend_name: str | None = None) -> None:
    jobs = [j for j in q.list_jobs() if not j.done]
    if not jobs:
        console.print("Queue is empty.")
        return
    chosen, note = choose_backend(backend_name)
    if note:
        console.print(f"[yellow]{note}[/]")
    if not chosen.is_available():
        console.print(f"[red]{chosen.name} backend unreachable; nothing sent.[/]")
        raise typer.Exit(1)
    console.print(f"Sending {len(jobs)} queued job(s) via {chosen.name}...")
    try:
        outcomes = q.drain(render=render_result, backend=chosen)
    except TypeError as e:
        if not _is_missing_credentials(e):
            raise
        console.print(NO_CREDS_MSG)
        raise typer.Exit(2)
    for job, result, err in outcomes:
        if result:
            console.print(f"[green]done[/] {job.name}: {escape(result.diagnosis.causes[0].title) if result.diagnosis.causes else 'no causes'}")
        else:
            console.print(f"[red]failed[/] {job.name}: {escape(err or '')}")


@app.command()
def queue_list():
    """Show queued and completed jobs."""
    jobs = q.list_jobs()
    if not jobs:
        console.print("Queue is empty.")
        return
    t = Table(title=str(config.QUEUE_DIR))
    t.add_column("Job")
    t.add_column("Status")
    for j in jobs:
        err = j.path / "error.txt"
        status = "done" if j.done else ("error: " + err.read_text().strip()[:80] if err.exists() else "pending")
        t.add_row(j.name, status)
    console.print(t)


@app.command()
def demo(
    offline: bool = typer.Option(False, help="Show the offline report instead of calling the API."),
    backend: str | None = typer.Option(None, help="'claude' or 'ollama'."),
    local_model: str | None = typer.Option(None, help="Ollama model name."),
    symptoms: str = typer.Option(
        "Rough idle and a shake at stoplights, worse when cold. Check engine light flashed once on the highway then stayed on.",
    ),
):
    """Ingest the seed knowledge and diagnose the bundled P0301 misfire fixture."""
    store = _store()
    results = ingest_path(store, config.SEED_KNOWLEDGE_DIR)
    added = sum(n for _, n, _ in results)
    console.print(f"[dim]seed knowledge: {added} new chunks, {store.count_chunks()} total[/]")
    fixture = config.FIXTURES_DIR / "p0301_misfire.json"
    diagnose(
        snapshot=None, fixture=fixture, port=None, symptoms=symptoms, audio=None, image=[], video=None,
        offline=offline, queue=False, drain=False, no_vin=False, strict=False, out=None, slow=False,
        backend=backend, local_model=local_model,
    )


if __name__ == "__main__":
    app()
