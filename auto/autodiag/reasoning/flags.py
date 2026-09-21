"""Derived observations from PID values.

Computed in code so retrieval and the model both reason from the same facts,
and so the offline report has something more useful than raw numbers.
Thresholds are generic; a manufacturer table could override them later.
"""

from __future__ import annotations

from ..models import Flag, VehicleSnapshot

FUEL_TRIM_WARN = 10.0  # percent
THERMOSTAT_MIN_C = 75.0
WARM_RUNTIME_S = 300


def _num(snapshot: VehicleSnapshot, name: str, section: str = "live") -> float | None:
    p = snapshot.pid(name, section)
    if p is None or not isinstance(p.value, (int, float)):
        return None
    return float(p.value)


def compute_flags(snapshot: VehicleSnapshot) -> list[Flag]:
    flags: list[Flag] = []
    for section in ("live", "freeze_frame"):
        label = "live" if section == "live" else "freeze frame"
        flags.extend(_fuel_trim_flags(snapshot, section, label))
        flags.extend(_thermal_flags(snapshot, section, label))
        flags.extend(_airflow_flags(snapshot, section, label))
        flags.extend(_o2_flags(snapshot, section, label))
        flags.extend(_electrical_flags(snapshot, section, label))
    flags.extend(_dtc_pattern_flags(snapshot))
    flags.extend(_readiness_flags(snapshot))
    return flags


def _fuel_trim_flags(s: VehicleSnapshot, section: str, label: str) -> list[Flag]:
    out = []
    for bank in (1, 2):
        stft = _num(s, f"SHORT_FUEL_TRIM_{bank}", section)
        ltft = _num(s, f"LONG_FUEL_TRIM_{bank}", section)
        if ltft is None:
            continue
        total = ltft + (stft or 0.0)
        if ltft > FUEL_TRIM_WARN or total > FUEL_TRIM_WARN + 5:
            out.append(
                Flag(
                    name=f"lean_bank{bank}_{section}",
                    severity="warn",
                    detail=(
                        f"Bank {bank} fuel trim positive ({label}): LTFT {ltft:+.1f}%"
                        + (f", STFT {stft:+.1f}%" if stft is not None else "")
                        + ". ECM is adding fuel, consistent with a lean condition (unmetered air, low fuel delivery, or MAF under-reporting)."
                    ),
                    pids=[f"LONG_FUEL_TRIM_{bank}", f"SHORT_FUEL_TRIM_{bank}"],
                    search_terms="lean fuel trim positive vacuum leak unmetered air fuel pressure MAF",
                )
            )
        elif ltft < -FUEL_TRIM_WARN or total < -(FUEL_TRIM_WARN + 5):
            out.append(
                Flag(
                    name=f"rich_bank{bank}_{section}",
                    severity="warn",
                    detail=(
                        f"Bank {bank} fuel trim negative ({label}): LTFT {ltft:+.1f}%"
                        + (f", STFT {stft:+.1f}%" if stft is not None else "")
                        + ". ECM is removing fuel, consistent with a rich condition (leaking injector, high fuel pressure, MAF over-reporting, restricted intake)."
                    ),
                    pids=[f"LONG_FUEL_TRIM_{bank}", f"SHORT_FUEL_TRIM_{bank}"],
                    search_terms="rich fuel trim negative leaking injector fuel pressure regulator MAF",
                )
            )
    # Bank split: one bank lean and the other not suggests a bank-specific cause.
    l1, l2 = _num(s, "LONG_FUEL_TRIM_1", section), _num(s, "LONG_FUEL_TRIM_2", section)
    if l1 is not None and l2 is not None and abs(l1 - l2) > FUEL_TRIM_WARN:
        out.append(
            Flag(
                name=f"bank_split_{section}",
                severity="warn",
                detail=f"Fuel trims differ between banks ({label}): B1 LTFT {l1:+.1f}%, B2 LTFT {l2:+.1f}%. Points to a bank-specific cause (injector, intake gasket, O2 sensor) rather than a common one (MAF, fuel pump).",
                pids=["LONG_FUEL_TRIM_1", "LONG_FUEL_TRIM_2"],
                search_terms="bank specific lean intake manifold gasket injector oxygen sensor",
            )
        )
    return out


def _thermal_flags(s: VehicleSnapshot, section: str, label: str) -> list[Flag]:
    out = []
    ect = _num(s, "COOLANT_TEMP", section)
    runtime = _num(s, "RUN_TIME", section)
    if ect is not None and ect < THERMOSTAT_MIN_C and (runtime is None or runtime > WARM_RUNTIME_S):
        out.append(
            Flag(
                name=f"coolant_low_{section}",
                severity="warn",
                detail=f"Coolant temperature {ect:.0f} degC ({label}) is below normal thermostat regulating range" + (f" after {runtime:.0f}s run time" if runtime else "") + ". Consistent with a stuck-open thermostat or an ECT sensor reading low.",
                pids=["COOLANT_TEMP", "RUN_TIME"],
                search_terms="thermostat stuck open coolant temperature sensor P0128 warm up",
            )
        )
    if ect is not None and ect > 110:
        out.append(
            Flag(
                name=f"coolant_high_{section}",
                severity="warn",
                detail=f"Coolant temperature {ect:.0f} degC ({label}) is high. Risk of overheating damage.",
                pids=["COOLANT_TEMP"],
                search_terms="overheating coolant temperature high cooling fan thermostat stuck closed",
            )
        )
    iat = _num(s, "INTAKE_TEMP", section)
    if iat is not None and (iat < -30 or iat > 80):
        out.append(
            Flag(
                name=f"iat_implausible_{section}",
                severity="warn",
                detail=f"Intake air temperature {iat:.0f} degC ({label}) is implausible; suspect IAT sensor or wiring.",
                pids=["INTAKE_TEMP"],
                search_terms="intake air temperature sensor circuit",
            )
        )
    return out


def _airflow_flags(s: VehicleSnapshot, section: str, label: str) -> list[Flag]:
    out = []
    rpm = _num(s, "RPM", section)
    maf = _num(s, "MAF", section)
    mapk = _num(s, "INTAKE_PRESSURE", section)
    speed = _num(s, "SPEED", section)
    if rpm is not None and maf is not None and 500 < rpm < 1200 and (speed is None or speed < 5):
        # Rough rule of thumb for a warm 4-cylinder at idle: ~1 g/s per liter displacement.
        if maf < 1.5:
            out.append(
                Flag(
                    name=f"maf_low_idle_{section}",
                    severity="warn",
                    detail=f"MAF {maf:.1f} g/s at {rpm:.0f} rpm idle ({label}) is low for most engines. Suspect MAF under-reporting or a large leak downstream of the sensor.",
                    pids=["MAF", "RPM"],
                    search_terms="MAF sensor low reading contaminated vacuum leak intake",
                )
            )
        elif maf > 8:
            out.append(
                Flag(
                    name=f"maf_high_idle_{section}",
                    severity="warn",
                    detail=f"MAF {maf:.1f} g/s at {rpm:.0f} rpm idle ({label}) is high; suspect MAF over-reporting or high idle.",
                    pids=["MAF", "RPM"],
                    search_terms="MAF sensor high reading idle",
                )
            )
    if mapk is not None and rpm is not None and 500 < rpm < 1200 and (speed is None or speed < 5):
        # Warm idle at sea level is 30-40 kPa; above 45 kPa vacuum is weak.
        if mapk > 45:
            out.append(
                Flag(
                    name=f"map_high_idle_{section}",
                    severity="warn",
                    detail=f"Manifold pressure {mapk:.0f} kPa at idle ({label}) indicates weak vacuum. Consistent with a vacuum leak, late cam timing, low compression, or a restricted exhaust.",
                    pids=["INTAKE_PRESSURE", "RPM"],
                    search_terms="low engine vacuum idle vacuum leak camshaft timing compression exhaust restriction",
                )
            )
        else:
            out.append(
                Flag(
                    name=f"map_normal_idle_{section}",
                    severity="info",
                    detail=f"Manifold pressure {mapk:.0f} kPa at idle ({label}) indicates normal engine vacuum, which argues against a large vacuum leak or mechanical breathing problem.",
                    pids=["INTAKE_PRESSURE", "RPM"],
                )
            )
    return out


def _o2_flags(s: VehicleSnapshot, section: str, label: str) -> list[Flag]:
    out = []
    up = _num(s, "O2_B1S1", section)
    down = _num(s, "O2_B1S2", section)
    if up is not None and (up < 0.1 or up > 0.85):
        out.append(
            Flag(
                name=f"o2_b1s1_extreme_{section}",
                severity="warn",
                detail=f"Upstream O2 B1S1 at {up:.2f} V ({label}) is near its rail; sensor may be stuck or the mixture is far from stoichiometric.",
                pids=["O2_B1S1"],
                search_terms="oxygen sensor stuck lean rich upstream",
            )
        )
    if down is not None and down > 0.6:
        out.append(
            Flag(
                name=f"o2_b1s2_high_{section}",
                severity="info",
                detail=f"Downstream O2 B1S2 steady at {down:.2f} V ({label}) suggests the catalyst is storing oxygen normally.",
                pids=["O2_B1S2"],
            )
        )
    return out


def _electrical_flags(s: VehicleSnapshot, section: str, label: str) -> list[Flag]:
    v = _num(s, "CONTROL_MODULE_VOLTAGE", section)
    rpm = _num(s, "RPM", section)
    if v is not None and rpm and rpm > 500 and (v < 13.0 or v > 15.0):
        return [
            Flag(
                name=f"charging_voltage_{section}",
                severity="warn",
                detail=f"Module voltage {v:.1f} V with engine running ({label}) is outside the normal 13.5 to 14.8 V charging range.",
                pids=["CONTROL_MODULE_VOLTAGE"],
                search_terms="charging system alternator voltage low high",
            )
        ]
    return []


def _dtc_pattern_flags(s: VehicleSnapshot) -> list[Flag]:
    codes = {d.code for d in s.dtcs}
    out = []
    cyl = sorted(c for c in codes if c.startswith("P030") and c != "P0300")
    if len(cyl) == 1:
        out.append(
            Flag(
                name="single_cylinder_misfire",
                severity="info",
                detail=f"Only one cylinder-specific misfire code ({cyl[0]}). Cylinder-specific causes (coil, plug, injector, compression on that cylinder) outrank common causes (fuel pressure, vacuum leak, MAF).",
                pids=[cyl[0]],
                search_terms="single cylinder misfire coil spark plug injector swap test compression",
            )
        )
    elif len(cyl) > 1:
        out.append(
            Flag(
                name="multi_cylinder_misfire",
                severity="warn",
                detail=f"Multiple cylinder-specific misfire codes ({', '.join(cyl)}). Look for a shared cause: fuel delivery, vacuum leak, ignition module, cam timing, or a pattern (same bank, adjacent cylinders suggests head gasket).",
                pids=cyl,
                search_terms="multiple cylinder misfire fuel pressure vacuum leak head gasket adjacent cylinders",
            )
        )
    if ("P0171" in codes or "P0174" in codes) and any(c.startswith("P030") for c in codes):
        out.append(
            Flag(
                name="lean_plus_misfire",
                severity="warn",
                detail="Lean code together with misfire code(s). A lean mixture can itself cause misfires; resolve the lean condition first.",
                pids=sorted(codes),
                search_terms="lean misfire vacuum leak fuel pressure",
            )
        )
    return out


def _readiness_flags(s: VehicleSnapshot) -> list[Flag]:
    incomplete = [k for k, v in s.readiness.items() if v == "incomplete"]
    dist = _num(s, "DISTANCE_SINCE_DTC_CLEAR")
    out = []
    if incomplete and dist is not None and dist < 100:
        out.append(
            Flag(
                name="recent_clear",
                severity="info",
                detail=f"Codes were cleared {dist:.0f} km ago and monitors {', '.join(incomplete)} have not completed. Some faults may not have re-set yet.",
                pids=["DISTANCE_SINCE_DTC_CLEAR"],
            )
        )
    return out
