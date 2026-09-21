"""Rules-only report built from local data. No API call. See DESIGN.md 4.7."""

from __future__ import annotations

from collections import defaultdict

from ..models import Chunk, Flag, VehicleSnapshot


def group_chunks_by_dtc(snapshot: VehicleSnapshot, chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Attach each chunk to the DTCs whose code appears in its text; else 'general'."""
    groups: dict[str, list[Chunk]] = defaultdict(list)
    codes = [d.code for d in snapshot.dtcs]
    for c in chunks:
        upper = c.text.upper()
        hit = False
        for code in codes:
            if code in upper:
                groups[code].append(c)
                hit = True
        if not hit:
            groups["general"].append(c)
    return groups


def render_offline_report(snapshot: VehicleSnapshot, flags: list[Flag], chunks: list[Chunk], max_chunk_chars: int = 900) -> str:
    lines: list[str] = []
    lines.append("OFFLINE REPORT (rules only)")
    lines.append(
        "This report was produced without the diagnostic model. It lists codes, derived flags, "
        "and matching manual excerpts. It contains no ranking, probabilities, or cross-evidence reasoning. "
        "Run `autodiag diagnose --drain` when online, or `--queue` to save this job."
    )
    lines.append("")

    v = snapshot.vehicle
    veh = " ".join(str(x) for x in (v.year, v.make, v.model, v.engine) if x)
    lines.append(f"Vehicle: {veh or 'unknown'}" + (f"  VIN {v.vin}" if v.vin else "") + (f"  {v.mileage} mi" if v.mileage else ""))
    lines.append(f"Source: {snapshot.source}")
    lines.append("")

    lines.append("== Trouble codes ==")
    if not snapshot.dtcs:
        lines.append("  none stored")
    for d in snapshot.dtcs:
        lines.append(f"  {d.code} [{d.status}]  {d.description}")
    lines.append("")

    lines.append("== Derived flags ==")
    if not flags:
        lines.append("  none")
    for f in flags:
        lines.append(f"  [{f.severity}] {f.detail}")
        vals = []
        # Flags computed from the freeze frame carry that suffix; show those values first.
        sections = ("freeze_frame", "live") if f.name.endswith("_freeze_frame") else ("live", "freeze_frame")
        for name in f.pids:
            p = snapshot.pid(name, sections[0]) or snapshot.pid(name, sections[1])
            if p is not None:
                vals.append(f"{name}={p.value}{p.unit}")
        if vals:
            lines.append(f"         raw: {', '.join(vals)}")
    lines.append("")

    lines.append("== Readiness monitors ==")
    if not snapshot.readiness:
        lines.append("  not available")
    for k, val in snapshot.readiness.items():
        lines.append(f"  {k}: {val}")
    lines.append("")

    lines.append("== Matching service knowledge ==")
    if not chunks:
        lines.append("  no excerpts matched; ingest manuals with `autodiag ingest`")
    groups = group_chunks_by_dtc(snapshot, chunks)
    order = [d.code for d in snapshot.dtcs] + ["general"]
    for key in order:
        items = groups.get(key)
        if not items:
            continue
        lines.append(f"-- {key} --")
        for c in items:
            loc = c.location()
            lines.append(f"  [{c.handle}] {c.title}" + (f" ({loc})" if loc else ""))
            body = c.text.strip()
            if len(body) > max_chunk_chars:
                body = body[:max_chunk_chars].rstrip() + " ..."
            for ln in body.splitlines():
                lines.append("      " + ln)
            lines.append("")
    return "\n".join(lines)
