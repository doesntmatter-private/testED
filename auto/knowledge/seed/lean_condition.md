# Lean Condition Workflow (P0171, P0174)

## Overview

P0171 (Bank 1) and P0174 (Bank 2) set when the engine control module has added as much fuel as it is allowed to through fuel trim and the oxygen sensor still reports lean. The module has run out of correction.

The key data are the short term and long term fuel trims and how they change with engine speed and load. Fuel trim is the percentage of extra fuel the module is adding (positive) or removing (negative).

## Interpreting fuel trim by condition

- High positive trim at idle that falls toward zero at 2500 rpm: unmetered air entering after the mass air flow sensor, a vacuum leak. At idle the leak is a large fraction of total airflow; at higher airflow it becomes negligible. This is the classic vacuum leak signature.
- High positive trim at all speeds and loads, roughly constant: the mass air flow sensor is under-reporting airflow (contaminated or failing), or fuel pressure is low across the board.
- Positive trim that grows with load or speed: fuel delivery cannot keep up. Suspect a weak fuel pump, restricted filter, or restricted injectors.
- Positive trim on one bank only with a normal other bank: the cause is bank-specific. Suspect an intake manifold gasket leak on that bank, an injector on that bank, an exhaust leak ahead of that bank's oxygen sensor, or the oxygen sensor itself.
- Both banks equally positive: the cause is common to both. Suspect the mass air flow sensor, fuel pressure, PCV system, or a leak at the intake boot or throttle body.

## Step 1: Fuel trim survey

1. Warm the engine fully.
2. Record short and long term fuel trims at idle, at 1500 rpm in neutral, and at 2500 rpm in neutral.
3. If available, record trims during a steady cruise at 50 to 60 mph.
4. Use the table above to classify the pattern before touching anything.

Expected time: 10 minutes. Tools: scan tool with live data.

## Step 2: Vacuum leak test

1. Inspect visually: intake boot cracks, disconnected or split vacuum hoses, PCV valve and hose, brake booster hose, dipstick seal, oil filler cap seal.
2. Smoke test: introduce smoke into the intake with the engine off and look for where it escapes. This is the definitive test.
3. If no smoke machine is available, with the engine idling spray a small amount of an approved intake cleaner or propane near suspect joints and watch for a change in engine speed or fuel trim. Use caution around hot exhaust and ignition sources.
4. Check manifold vacuum with a gauge: a warm engine at idle at sea level should show 17 to 22 inches of mercury (roughly 30 to 40 kPa manifold absolute pressure). Lower vacuum with a lean code supports a leak.

## Step 3: Mass air flow sensor plausibility

1. At idle, a warm engine flows roughly 1 gram per second per liter of displacement. A 2.5 liter engine should show about 2 to 3.5 grams per second. A reading well below this with a lean code suggests under-reporting.
2. Snap the throttle open: the reading should rise quickly and smoothly. Hesitation or a spike indicates a contaminated or failing sensor element.
3. Compare the calculated airflow from manifold pressure and engine speed to the sensor reading if the scan tool supports volumetric efficiency.
4. Cleaning with a dedicated mass air flow sensor cleaner is a low-cost first step if the element is visibly dirty, but a cleaned sensor that reads the same is not the cause.

## Step 4: Fuel pressure test

1. Connect a fuel pressure gauge at the rail test port (or read the fuel rail pressure sensor on direct injection engines).
2. Record pressure at key-on engine-off, at idle, and while snapping the throttle or under load. Pressure should hold within specification and not sag under load.
3. Pressure that is low at idle points to the pump, filter, or regulator. Pressure that is fine at idle but drops under load points to a weak pump or restricted filter.
4. Turn the engine off and watch the pressure hold. A fast drop indicates a leaking injector, a leaking regulator, or a pump check valve.

Caution: fuel systems hold pressure after shutdown. Relieve pressure and have absorbent material ready before opening any fitting.

## Step 5: Exhaust leak ahead of the sensor

An exhaust leak before the upstream oxygen sensor pulls in outside air between exhaust pulses, which the sensor reads as lean. Look for soot marks at the manifold gasket, flex pipe, and sensor threads. Listen for ticking that changes with engine speed on a cold start.

## Step 6: Oxygen sensor

A lazy or skewed upstream sensor can report lean when the mixture is not. Watch the sensor voltage switch above and below 0.45 volts several times per second at 2000 rpm. A sensor that stays low or switches slowly is suspect. Confirm with an exhaust gas analyzer or by comparing to the downstream sensor when the catalyst is bypassed on a scan tool that supports it.

## Common causes by frequency

Vacuum leaks (intake boot, PCV, manifold gasket) are the most common cause of a lean code at idle. Mass air flow sensor contamination is the most common cause of a constant lean offset. Fuel pressure problems are less common but more expensive to miss because they can damage the catalyst through misfire.
