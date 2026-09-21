# Coolant Temperature Workflow (P0128, P0116, P0117, P0118)

## Overview

P0128 sets when the engine coolant temperature does not reach the thermostat's regulating temperature within the time the module expects, given ambient temperature and how long the engine has run. P0116, P0117 and P0118 are sensor circuit codes: range or performance, low input, and high input respectively.

The thermostat regulating temperature on most engines is between 82 and 95 degrees Celsius (180 to 203 degrees Fahrenheit). A warm engine that idles at 70 degrees Celsius on a mild day is not reaching regulation.

## Step 1: Confirm with live data

1. From a cold start, graph coolant temperature and intake air temperature. Both should read close to ambient before start; a coolant sensor reading far from the intake air sensor on a cold engine is suspect.
2. Drive normally for 15 minutes. Coolant temperature should climb steadily and level off at the thermostat rating.
3. Coolant that climbs slowly and levels off well below the rating, especially in cold weather or at highway speed, indicates a thermostat stuck open or missing.
4. Coolant that reads correctly on the scan tool but the dash gauge disagrees points to the gauge sender (a separate sensor on many vehicles).

## Step 2: Physical check of the thermostat

1. With the engine cold, feel the upper radiator hose while the engine warms. It should stay cool for several minutes and then become hot quickly when the thermostat opens. A hose that warms immediately from a cold start indicates a thermostat stuck open.
2. Use an infrared thermometer on the thermostat housing and the radiator inlet to see when flow starts.
3. If the thermostat is easily accessible, remove it and test in a pan of water with a thermometer. It should be closed at room temperature and fully open within about 10 degrees of its rating.

## Step 3: Sensor circuit checks (P0116, P0117, P0118)

1. P0117 (low input, high temperature reading): the sensor circuit is shorted to ground or the sensor is shorted. Disconnect the sensor; the reading should go to the cold extreme. If it does not, the wiring is shorted.
2. P0118 (high input, low or minus 40 reading): the circuit is open. Disconnect the sensor and jumper the two terminals; the reading should go to the hot extreme. If it does not, the wiring or module is open.
3. Measure sensor resistance at a known temperature and compare to the manufacturer's table. Most are negative temperature coefficient thermistors: resistance falls as temperature rises.
4. Check the connector for corrosion and coolant intrusion.

## Step 4: Other causes of slow warm-up

Low coolant level (air pocket at the sensor), a cooling fan running constantly, a stuck-open coolant bypass or heater control valve, and very short trips in cold weather can all produce P0128 with a good thermostat. Check coolant level and fan operation before condemning the thermostat.

## Consequences of ignoring P0128

An engine running cold uses more fuel, wears faster, and may not enter closed-loop fuel control promptly. It is not a safety issue but should be repaired.
