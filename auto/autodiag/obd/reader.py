"""OBD-II readers.

`PythonOBDReader` talks to a real ELM327-style adapter via python-OBD.
`SimulatedReader` replays a JSON fixture so the whole pipeline runs on a desk.
Both produce the same `VehicleSnapshot`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from ..models import DTC, PIDValue, VehicleInfo, VehicleSnapshot
from .dtc_db import describe

# Live PIDs most useful for engine troubleshooting. Names match python-OBD's
# `obd.commands` attribute names.
ENGINE_PIDS: list[tuple[str, str]] = [
    ("RPM", "rpm"),
    ("ENGINE_LOAD", "%"),
    ("COOLANT_TEMP", "degC"),
    ("INTAKE_TEMP", "degC"),
    ("SHORT_FUEL_TRIM_1", "%"),
    ("LONG_FUEL_TRIM_1", "%"),
    ("SHORT_FUEL_TRIM_2", "%"),
    ("LONG_FUEL_TRIM_2", "%"),
    ("MAF", "g/s"),
    ("INTAKE_PRESSURE", "kPa"),
    ("THROTTLE_POS", "%"),
    ("TIMING_ADVANCE", "deg"),
    ("O2_B1S1", "V"),
    ("O2_B1S2", "V"),
    ("O2_S1_WR_VOLTAGE", "V"),
    ("FUEL_PRESSURE", "kPa"),
    ("FUEL_RAIL_PRESSURE_DIRECT", "kPa"),
    ("SPEED", "km/h"),
    ("CONTROL_MODULE_VOLTAGE", "V"),
    ("CATALYST_TEMP_B1S1", "degC"),
    ("EVAP_VAPOR_PRESSURE", "Pa"),
    ("FUEL_LEVEL", "%"),
    ("RUN_TIME", "s"),
    ("DISTANCE_SINCE_DTC_CLEAR", "km"),
    ("WARMUPS_SINCE_DTC_CLEAR", "count"),
]

READINESS_TESTS = (
    "MISFIRE_MONITORING",
    "FUEL_SYSTEM_MONITORING",
    "COMPONENT_MONITORING",
    "CATALYST_MONITORING",
    "EVAPORATIVE_SYSTEM_MONITORING",
    "OXYGEN_SENSOR_MONITORING",
    "OXYGEN_SENSOR_HEATER_MONITORING",
    "EGR_VVT_SYSTEM_MONITORING",
)


class OBDReader(Protocol):
    def read(self) -> VehicleSnapshot: ...
    def close(self) -> None: ...


def _pid_value(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


class SimulatedReader:
    """Replays a JSON fixture. See fixtures/*.json for the shape.

    Fixture keys: vehicle, dtcs, freeze_frame, live, readiness. An optional
    `expected` block (true cause, confirming test) is ignored here and used
    by the eval harness.
    """

    def __init__(self, fixture: str | Path):
        self.path = Path(fixture)
        self.data = json.loads(self.path.read_text())

    def read(self) -> VehicleSnapshot:
        d = self.data
        dtcs = [
            DTC(
                code=c["code"].upper(),
                description=c.get("description") or describe(c["code"]),
                status=c.get("status", "stored"),
            )
            for c in d.get("dtcs", [])
        ]

        def pids(section: str) -> list[PIDValue]:
            return [
                PIDValue(name=k, value=_pid_value(v.get("value")), unit=v.get("unit", ""))
                for k, v in d.get(section, {}).items()
            ]

        return VehicleSnapshot(
            vehicle=VehicleInfo(**d.get("vehicle", {})),
            dtcs=dtcs,
            freeze_frame=pids("freeze_frame"),
            live=pids("live"),
            readiness={k: str(v) for k, v in d.get("readiness", {}).items()},
            source=f"simulated:{self.path.name}",
        )

    def close(self) -> None:
        return None


class PythonOBDReader:
    """Real adapter via python-OBD (ELM327 over USB/Bluetooth/WiFi)."""

    def __init__(self, port: str | None = None, baudrate: int | None = None, fast: bool = True):
        import obd  # lazy: the simulator should not need the serial stack

        self._obd = obd
        self.conn = obd.OBD(portstr=port, baudrate=baudrate, fast=fast, timeout=5)
        if not self.conn.is_connected():
            raise ConnectionError(
                f"Could not connect to OBD adapter (port={port!r}). "
                "Check the adapter, ignition ON, and the serial port."
            )

    def _query(self, name: str):
        cmd = getattr(self._obd.commands, name, None)
        if cmd is None or not self.conn.supports(cmd):
            return None
        resp = self.conn.query(cmd)
        if resp.is_null():
            return None
        return resp.value

    def _dtcs(self, cmd_name: str, status: str) -> list[DTC]:
        val = self._query(cmd_name)
        if not val:
            return []
        return [
            DTC(code=code, description=text or describe(code), status=status)  # type: ignore[arg-type]
            for code, text in val
        ]

    def _pids(self, prefix: str) -> list[PIDValue]:
        out: list[PIDValue] = []
        for name, unit in ENGINE_PIDS:
            val = self._query(f"{prefix}{name}")
            if val is None:
                continue
            magnitude = getattr(val, "magnitude", val)
            out.append(
                PIDValue(
                    name=name,
                    value=_pid_value(magnitude),
                    unit=str(getattr(val, "units", unit)),
                )
            )
        return out

    def read(self) -> VehicleSnapshot:
        dtcs = self._dtcs("GET_DTC", "stored") + self._dtcs("GET_CURRENT_DTC", "pending")
        # Permanent DTCs (Mode 0A) are out of scope for M1.

        live = self._pids("")
        freeze = self._pids("DTC_")
        ff_dtc = self._query("DTC_FREEZE_DTC")
        if ff_dtc:
            freeze.insert(0, PIDValue(name="FREEZE_DTC", value=str(ff_dtc[0]), unit=""))

        readiness: dict[str, str] = {}
        status = self._query("STATUS")
        if status is not None:
            readiness["MIL"] = "on" if getattr(status, "MIL", False) else "off"
            readiness["DTC_count"] = str(getattr(status, "DTC_count", ""))
            for test_name in READINESS_TESTS:
                test = getattr(status, test_name, None)
                if test is not None and getattr(test, "available", False):
                    readiness[test_name] = "complete" if getattr(test, "complete", False) else "incomplete"

        vin_val = self._query("VIN")
        if isinstance(vin_val, bytes):
            vin: str | None = vin_val.decode(errors="ignore")
        else:
            vin = str(vin_val) if vin_val else None

        return VehicleSnapshot(
            vehicle=VehicleInfo(vin=vin),
            dtcs=dtcs,
            freeze_frame=freeze,
            live=live,
            readiness=readiness,
            source=f"obd:{self.conn.port_name()}",
        )

    def close(self) -> None:
        self.conn.close()


def open_reader(port: str | None = None, fixture: str | Path | None = None, fast: bool = True) -> OBDReader:
    """Return a simulated reader if a fixture is given, else a live adapter."""
    if fixture:
        return SimulatedReader(fixture)
    return PythonOBDReader(port=port, fast=fast)
