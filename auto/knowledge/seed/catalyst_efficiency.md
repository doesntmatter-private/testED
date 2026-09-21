# Catalyst Efficiency Workflow (P0420, P0430)

## Overview

P0420 (Bank 1) and P0430 (Bank 2) set when the downstream oxygen sensor's signal looks too much like the upstream sensor's signal. A healthy catalyst stores oxygen, so the downstream sensor should be relatively steady, typically between 0.6 and 0.8 volts, while the upstream sensor switches rapidly. When the downstream sensor starts switching along with the upstream one, the converter is not storing oxygen.

The code does not by itself mean the catalyst has failed. Anything that changes the exhaust gas mixture or the sensor readings can set it.

## Step 1: Rule out upstream causes first

Do not replace a catalyst until these are checked:

1. Misfire codes, current or history. Misfires dump unburned fuel into the catalyst and can both damage it and set P0420 while the misfire is present.
2. Lean or rich codes and fuel trims. A rich mixture can saturate the catalyst; a lean mixture can make it run cool and inefficient.
3. Exhaust leaks between the upstream and downstream sensors, or at the downstream sensor threads. Outside air at the downstream sensor mimics a dead catalyst.
4. Engine oil or coolant consumption. Both poison the catalyst. Ask about oil top-ups and check for coolant loss.
5. Age and quality of the downstream sensor. A slow downstream sensor can produce the same signature as a bad catalyst.

## Step 2: Compare upstream and downstream sensor waveforms

1. Warm the engine and the catalyst fully: at least 10 minutes of driving or 2500 rpm in neutral for several minutes.
2. Graph both sensors on the scan tool at 2000 rpm steady.
3. Healthy: upstream switching 0.1 to 0.9 volts several times per second, downstream mostly steady between 0.6 and 0.8 volts with slow drift.
4. Failed: downstream following the upstream switching, or downstream steadily low.

## Step 3: Catalyst temperature check

With an infrared thermometer, measure the pipe temperature just before and just after the catalyst at a fully warm 2000 rpm. The outlet should be at least 50 to 100 degrees Fahrenheit hotter than the inlet. An outlet that is the same temperature or cooler indicates no catalytic activity. This test is quick but affected by airflow and pipe shielding.

## Step 4: Backpressure test

A melted or plugged catalyst restricts exhaust flow. Remove the upstream oxygen sensor and install a backpressure gauge. At idle, pressure should be below 1.5 psi; at 2500 rpm below 3 psi. Higher readings indicate a restriction. Loss of power at high rpm and a rattling noise from the converter also suggest a broken substrate.

## Step 5: Fuel quality and aftermarket parts

Aftermarket universal converters often set P0420 on vehicles calibrated for original equipment converters, even when new. Check for recent exhaust work. Fuel additives claiming to fix P0420 rarely do.

## Decision

Replace the catalyst only after Steps 1 through 4 point to it and any root cause (misfire, oil consumption, rich or lean condition) has been fixed. A new catalyst installed on an engine that is still misfiring or burning oil will fail again.
