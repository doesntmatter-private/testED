# Airflow and Manifold Pressure Sensor Plausibility (P0101, P0102, P0106)

## Overview

The mass air flow sensor (MAF) measures the mass of air entering the engine in grams per second. The manifold absolute pressure sensor (MAP) measures pressure in the intake manifold in kilopascals. Many engines have both; the module cross-checks them. P0101 is MAF range or performance, P0102 is MAF low input, P0106 is MAP range or performance.

## Expected values

- MAP, engine off, key on: equals barometric pressure, about 101 kPa at sea level, falling roughly 1 kPa per 100 meters of altitude.
- MAP, warm idle: 30 to 40 kPa at sea level (17 to 22 inches of mercury vacuum). Higher pressure at idle means less vacuum: a leak, late valve timing, low compression, or a restricted exhaust.
- MAP, wide open throttle: approaches barometric pressure.
- MAF, warm idle: roughly 1 gram per second per liter of displacement. A 2.0 liter engine idles around 2 to 3 grams per second; a 5.7 liter around 5 to 8.
- MAF, snap throttle: rises quickly and smoothly, typically well above 100 grams per second on a wide open snap.

## Step 1: Cross-check MAF against MAP and RPM

If both sensors are present, calculate expected airflow from manifold pressure, engine speed, displacement and an assumed volumetric efficiency of about 75 percent at idle. A MAF reading more than 20 percent away from the calculated value is suspect. Scan tools with a volumetric efficiency test automate this.

## Step 2: MAF inspection and cleaning

1. Inspect the intake between the air filter and the MAF for a dirty or oiled filter, or an aftermarket oiled filter that has contaminated the element.
2. Inspect the element for dust, oil film, or debris.
3. Clean only with a dedicated MAF cleaner; never touch the element or use brake cleaner.
4. Retest. If the reading does not change, cleaning was not the fix.

## Step 3: Wiring and connector

Check the connector for corrosion and bent pins. Measure reference voltage and ground at the connector. A poor ground raises the signal at idle and can set a range code.

## Step 4: Air leaks between MAF and throttle

Any air entering after the MAF is unmeasured. Inspect the intake boot, clamps, and any hoses attached to the boot. Squeeze the boot with the engine idling and watch fuel trims and MAF; a change indicates a leak.

## Step 5: MAP sensor test

1. Apply vacuum with a hand pump and watch the MAP reading fall smoothly and proportionally.
2. Check the vacuum hose to the sensor for cracks, kinks, or oil contamination. Oil in the hose points to excessive blow-by or PCV problems.
3. Compare MAP at key on engine off to a known barometric pressure.

## Common causes by frequency

Contaminated MAF element, torn intake boot after the MAF, cracked MAP vacuum hose, aftermarket oiled air filter, wiring or connector faults.
