# Misfire Isolation Workflow (P0300 to P0308)

## Overview

Codes P0301 through P0308 identify a misfire on a specific cylinder (P0301 is cylinder 1, P0302 cylinder 2, and so on). P0300 is a random or multiple cylinder misfire. The misfire monitor detects small changes in crankshaft acceleration between firing events using the crankshaft position sensor.

A misfire code with a flashing check engine light indicates a catalyst-damaging misfire rate. Advise the owner to limit driving until repaired.

Every misfire needs three things to be present: spark, fuel, and compression, delivered at the correct time. Isolation is a process of deciding which of these is missing on the affected cylinder.

## Step 1: Read freeze frame and classify

Record the freeze frame stored with the misfire code. Note engine RPM, engine load, coolant temperature, vehicle speed, and fuel trims.

- Misfire at idle and low load only, warm engine: suspect ignition (coil, plug), a leaking injector, or a small vacuum leak near that cylinder's intake runner.
- Misfire under load or acceleration only: suspect ignition breaking down under cylinder pressure (worn plug, weak coil), or a restricted injector.
- Misfire only when cold: suspect a low-flow or leaking injector, carbon deposits, or a valve sealing problem that closes up when warm.
- Misfire with high positive fuel trims: resolve the lean condition first (see the lean condition workflow). A lean mixture misfires on its own.

## Step 2: Swap test (single-cylinder misfire only)

The swap test is the fastest way to separate an ignition component fault from a cylinder fault. It applies when exactly one cylinder-specific code is present.

1. With the engine off and cool, remove the ignition coil from the misfiring cylinder and swap it with the coil from a known-good cylinder. On coil-on-plug engines this takes a few minutes. Mark the coils.
2. Clear codes. Drive or idle under the same conditions recorded in the freeze frame until the misfire returns (typically 5 to 15 minutes).
3. Read codes again.
   - If the misfire code moved to the cylinder that now has the suspect coil: the coil is faulty. Replace the coil.
   - If the misfire stayed on the original cylinder: the coil is good. Proceed to swap the spark plug the same way.
   - If the spark plug swap also does not move the misfire: the cause is on the cylinder itself. Go to Step 4 (injector) and Step 5 (mechanical).

Expected time: 20 to 40 minutes including the drive cycle. Tools: basic hand tools, spark plug socket, scan tool to clear codes.

## Step 3: Inspect the spark plug

When the plug is out, read it:

- Wet with fuel: cylinder is not firing (ignition) or is flooding (leaking injector).
- White or blistered insulator: lean or overheating on that cylinder.
- Oily deposits: oil entering the chamber (valve seals, rings, PCV).
- Carbon fouled: rich, or excessive idling.
- Cracked insulator or worn electrode gap beyond specification: replace.

Compare the gap to specification. A gap opened by wear increases the voltage demand and causes misfire under load first.

## Step 4: Injector test

1. With the engine running, listen at each injector with a mechanic's stethoscope or long screwdriver. A clicking injector is being commanded. A silent injector on the misfiring cylinder points to the injector or its driver circuit.
2. Measure injector coil resistance and compare to specification and to the other injectors. An open or shorted injector is a clear fault.
3. Check the injector's balance or drop-in-pressure test if the scan tool supports it. A drop that is much smaller than the others indicates a restricted injector; much larger indicates a leaking injector.
4. Swap the injector with a known-good cylinder if the above is inconclusive and the injectors are easily accessible. If the misfire moves, replace the injector.

## Step 5: Mechanical checks

If ignition and fuel are good, the cylinder itself is suspect.

1. Compression test. Remove all spark plugs, disable fuel, crank each cylinder for the same number of strokes. Cylinders should be within 10 percent of each other. A low cylinder points to rings, valves, or a head gasket.
2. If compression is low, add a small amount of oil to the cylinder and retest (wet test). A large rise points to rings; no change points to valves or head gasket.
3. Cylinder leak-down test gives a more precise picture of where the leak is (listen at the intake, exhaust, oil filler, and radiator).
4. Two adjacent cylinders both low: suspect a head gasket between them.
5. Check valve clearance on engines with mechanical adjusters. A tight valve does not seal when warm and causes a misfire that gets worse as the engine heats up.

## Step 6: Random or multiple misfire (P0300) without a single-cylinder pattern

A shared cause is likely:

- Low fuel pressure (fuel pump, filter, regulator). Measure fuel pressure at idle and under load.
- Large vacuum leak (intake boot, PCV hose, brake booster hose, intake manifold gasket). Check manifold vacuum and listen for hissing.
- Ignition module or crank sensor signal noise.
- Camshaft timing off (stretched chain, jumped belt). Compare cam and crank signals or check cam timing marks.
- EGR valve stuck open at idle.
- Contaminated fuel.

## Common causes by frequency

For a single-cylinder misfire on a modern coil-on-plug engine, in rough order of frequency: ignition coil, spark plug, fuel injector, valve or compression problem, wiring or connector to the coil or injector, intake gasket leak near that runner.
