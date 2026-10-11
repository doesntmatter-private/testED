"""Caltrans Lane Closure System (LCS) client and site metadata.

LCS publishes every approved closure for the next 7 days plus all active closures, per district,
refreshed every 5 minutes, free of charge:
  https://cwwp2.dot.ca.gov/data/d{N}/lcs/lcsStatusD{NN}.json
Field reference: https://cwwp2.dot.ca.gov/documentation/lcs/lcs-field-description.htm

The selected closure is stored as <lane>.site.json next to the lane file and stamped into event JSON.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("laneguard.lcs")

LCS_URL = "https://cwwp2.dot.ca.gov/data/d{d}/lcs/lcsStatusD{d:02d}.json"
DISTRICTS = {
    1: "Eureka", 2: "Redding", 3: "Marysville / Sacramento", 4: "Bay Area", 5: "San Luis Obispo", 6: "Fresno",
    7: "Los Angeles", 8: "San Bernardino", 9: "Bishop", 10: "Stockton", 11: "San Diego", 12: "Orange County",
}
_cache: dict[int, tuple[float, list[dict]]] = {}
_cache_lock = threading.Lock()
CACHE_TTL_S = 300  # matches the feed refresh interval


def _s(v) -> str:
    return "" if v is None else str(v).strip()


def flatten(rec: dict) -> dict:
    """Flatten one LCS record into the fields Lane Guard cares about."""
    r = rec.get("lcs", rec)
    loc = r.get("location", {})
    b, e = loc.get("begin", {}), loc.get("end", {})
    c = r.get("closure", {})
    ts = c.get("closureTimestamp", {})
    def flag(name):
        blk = c.get(name, {})
        return _s(blk.get(f"is{name[0].upper()}{name[1:]}")) == "true", _s(blk.get(f"{name}Timestamp", {}).get(f"{name}Epoch"))
    is1097, t1097 = flag("code1097")
    is1098, t1098 = flag("code1098")
    is1022, t1022 = flag("code1022")
    status = "cancelled" if is1022 else "picked_up" if is1098 else "active" if is1097 else "planned"
    pm = lambda p, m, a: f"{p}{m}{(' ' + a) if a else ''}".strip()
    return {
        "index": _s(r.get("index")),
        "closureID": _s(c.get("closureID")),
        "logNumber": _s(c.get("logNumber")),
        "district": _s(b.get("beginDistrict")),
        "county": _s(b.get("beginCounty")),
        "route": _s(b.get("beginRoute")),
        "direction": _s(loc.get("travelFlowDirection")),
        "beginPostmile": pm(_s(b.get("beginPostmilePrefix")), _s(b.get("beginPostmile")), _s(b.get("beginAlignment"))),
        "endPostmile": pm(_s(e.get("endPostmilePrefix")), _s(e.get("endPostmile")), _s(e.get("endAlignment"))),
        "beginMilepost": _s(b.get("beginMilepost")),
        "endMilepost": _s(e.get("endMilepost")),
        "beginLocation": _s(b.get("beginLocationName")),
        "endLocation": _s(e.get("endLocationName")),
        "nearbyPlace": _s(b.get("beginNearbyPlace")),
        "beginLat": _float(b.get("beginLatitude")),
        "beginLon": _float(b.get("beginLongitude")),
        "endLat": _float(e.get("endLatitude")),
        "endLon": _float(e.get("endLongitude")),
        "facility": _s(c.get("facility")),
        "typeOfClosure": _s(c.get("typeOfClosure")),
        "typeOfWork": _s(c.get("typeOfWork")),
        "durationOfClosure": _s(c.get("durationOfClosure")),
        "lanesClosed": _s(c.get("lanesClosed")),
        "totalExistingLanes": _s(c.get("totalExistingLanes")),
        "estimatedDelay": _s(c.get("estimatedDelay")),
        "isCHINReportable": _s(c.get("isCHINReportable")) == "true",
        "start": f"{_s(ts.get('closureStartDate'))} {_s(ts.get('closureStartTime'))}".strip(),
        "end": f"{_s(ts.get('closureEndDate'))} {_s(ts.get('closureEndTime'))}".strip(),
        "startEpoch": _int(ts.get("closureStartEpoch")),
        "endEpoch": _int(ts.get("closureEndEpoch")),
        "status": status,
        "code1097Epoch": _int(t1097), "code1098Epoch": _int(t1098), "code1022Epoch": _int(t1022),
    }


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fetch_district(district: int, force: bool = False, timeout: float = 20.0) -> list[dict]:
    """Download and flatten one district feed, cached for CACHE_TTL_S."""
    if district not in DISTRICTS:
        raise ValueError(f"district must be 1..12, got {district}")
    with _cache_lock:
        hit = _cache.get(district)
        if hit and not force and time.time() - hit[0] < CACHE_TTL_S:
            return hit[1]
    url = LCS_URL.format(d=district)
    req = urllib.request.Request(url, headers={"User-Agent": "laneguard/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.load(resp)
    records = raw.get("data", raw if isinstance(raw, list) else [])
    flat = [flatten(r) for r in records]
    with _cache_lock:
        _cache[district] = (time.time(), flat)
    log.info("LCS district %d: %d closures", district, len(flat))
    return flat


def haversine_mi(lat1, lon1, lat2, lon2) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search(records: list[dict], q: str = "", status: str = "", lat: float | None = None, lon: float | None = None,
           radius_mi: float = 5.0, lane_only: bool = False, limit: int = 100) -> list[dict]:
    """Filter and rank closures. Nearest first when a position is given, else soonest start first."""
    ql = q.lower().strip()
    out = []
    for r in records:
        if status and r["status"] != status:
            continue
        if lane_only and r["typeOfClosure"] not in ("Lane", "One-Way Traffic", "Alternating", "Alternating Lanes", "Moving"):
            continue
        if ql and not any(ql in _s(r[k]).lower() for k in
                          ("route", "county", "nearbyPlace", "beginLocation", "endLocation", "typeOfWork", "closureID", "index")):
            continue
        d = None
        if lat is not None and lon is not None and r["beginLat"] is not None:
            d = min(haversine_mi(lat, lon, r["beginLat"], r["beginLon"]),
                    haversine_mi(lat, lon, r["endLat"], r["endLon"]) if r["endLat"] is not None else 1e9)
            if d > radius_mi:
                continue
        out.append({**r, "distance_mi": round(d, 2) if d is not None else None})
    if lat is not None:
        out.sort(key=lambda r: (r["distance_mi"] if r["distance_mi"] is not None else 1e9))
    else:
        out.sort(key=lambda r: (r["status"] != "active", r["startEpoch"] or 0))
    return out[:limit]


# ---- site metadata ---------------------------------------------------------

def site_path(lane_path: Path) -> Path:
    lane_path = Path(lane_path)
    return lane_path.with_name(lane_path.stem + ".site.json")


def load_site(lane_path: Path) -> dict:
    try:
        return json.loads(site_path(lane_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_site(lane_path: Path, site: dict) -> dict:
    site = {**site, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = site_path(lane_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(site, indent=2))
    return site


def event_stamp(site: dict) -> dict:
    """The subset of site metadata written into every event JSON."""
    if not site:
        return {}
    lcs = site.get("lcs", {})
    keys = ("index", "closureID", "logNumber", "district", "county", "route", "direction",
            "beginPostmile", "endPostmile", "typeOfWork", "typeOfClosure", "lanesClosed", "facility", "start", "end")
    return {
        "lcs": {k: lcs.get(k) for k in keys if k in lcs},
        "crew": site.get("crew", ""), "operator": site.get("operator", ""), "notes": site.get("notes", ""),
        "camera_lat": site.get("camera_lat"), "camera_lon": site.get("camera_lon"),
    }
