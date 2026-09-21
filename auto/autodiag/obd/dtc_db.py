"""Small built-in table of generic (SAE J2012) powertrain codes.

python-OBD ships a much larger table; this exists so the simulator and tests
work without it, and so we can enrich codes the adapter returns without text.
"""

GENERIC_DTCS: dict[str, str] = {
    "P0011": "'A' Camshaft Position - Timing Over-Advanced or System Performance (Bank 1)",
    "P0016": "Crankshaft Position - Camshaft Position Correlation (Bank 1 Sensor A)",
    "P0030": "HO2S Heater Control Circuit (Bank 1 Sensor 1)",
    "P0087": "Fuel Rail/System Pressure - Too Low",
    "P0101": "Mass or Volume Air Flow Circuit Range/Performance",
    "P0102": "Mass or Volume Air Flow Circuit Low Input",
    "P0106": "Manifold Absolute Pressure/Barometric Pressure Circuit Range/Performance",
    "P0113": "Intake Air Temperature Circuit High Input",
    "P0116": "Engine Coolant Temperature Circuit Range/Performance",
    "P0117": "Engine Coolant Temperature Circuit Low Input",
    "P0118": "Engine Coolant Temperature Circuit High Input",
    "P0121": "Throttle/Pedal Position Sensor/Switch A Circuit Range/Performance",
    "P0128": "Coolant Thermostat (Coolant Temperature Below Thermostat Regulating Temperature)",
    "P0131": "O2 Sensor Circuit Low Voltage (Bank 1 Sensor 1)",
    "P0133": "O2 Sensor Circuit Slow Response (Bank 1 Sensor 1)",
    "P0135": "O2 Sensor Heater Circuit (Bank 1 Sensor 1)",
    "P0171": "System Too Lean (Bank 1)",
    "P0172": "System Too Rich (Bank 1)",
    "P0174": "System Too Lean (Bank 2)",
    "P0175": "System Too Rich (Bank 2)",
    "P0300": "Random/Multiple Cylinder Misfire Detected",
    "P0301": "Cylinder 1 Misfire Detected",
    "P0302": "Cylinder 2 Misfire Detected",
    "P0303": "Cylinder 3 Misfire Detected",
    "P0304": "Cylinder 4 Misfire Detected",
    "P0305": "Cylinder 5 Misfire Detected",
    "P0306": "Cylinder 6 Misfire Detected",
    "P0325": "Knock Sensor 1 Circuit (Bank 1)",
    "P0335": "Crankshaft Position Sensor A Circuit",
    "P0340": "Camshaft Position Sensor A Circuit (Bank 1)",
    "P0351": "Ignition Coil A Primary/Secondary Circuit",
    "P0401": "Exhaust Gas Recirculation Flow Insufficient Detected",
    "P0420": "Catalyst System Efficiency Below Threshold (Bank 1)",
    "P0430": "Catalyst System Efficiency Below Threshold (Bank 2)",
    "P0440": "Evaporative Emission System",
    "P0442": "Evaporative Emission System Leak Detected (Small Leak)",
    "P0455": "Evaporative Emission System Leak Detected (Large Leak)",
    "P0456": "Evaporative Emission System Leak Detected (Very Small Leak)",
    "P0500": "Vehicle Speed Sensor A",
    "P0505": "Idle Air Control System",
    "P0506": "Idle Air Control System RPM Lower Than Expected",
    "P0507": "Idle Air Control System RPM Higher Than Expected",
    "P0520": "Engine Oil Pressure Sensor/Switch Circuit",
    "P0562": "System Voltage Low",
    "P0600": "Serial Communication Link",
    "P0700": "Transmission Control System (MIL Request)",
}


def describe(code: str, fallback: str = "") -> str:
    return GENERIC_DTCS.get(code.upper(), fallback or "Unknown code (not in built-in table)")
