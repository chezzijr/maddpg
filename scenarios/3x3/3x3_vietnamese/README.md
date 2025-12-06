# Vietnamese Traffic Scenario - 3x3 Grid

## Scenario Characteristics

### Network
- Grid size: 3×3 intersections
- Block length: 200m between intersections
- Lanes: 2 lanes per direction
- Traffic signals: Actuated control at all intersections
- Sublane model: ENABLED (for motorcycle lane-splitting)

### Vehicle Composition (Vietnamese Urban)
- **Motorcycles**: 70% (dominant mode)
- **Cars**: 20%
- **Trucks**: 5%
- **Buses**: 5%

### Time-of-Day Patterns (24-hour simulation)

#### Early Morning (00:00-06:00)
- Flow: ~100 veh/h per entry
- Truck ratio: 15% (freight delivery)
- Directional bias: Slight towards urban

#### Morning Rush (06:00-09:00) 🔥
- Flow: ~1200 veh/h per entry
- Heavy bias TO urban center (westbound)
- Minimal trucks
- Peak motorcycle usage

#### Midday (09:00-16:00)
- Flow: ~600 veh/h per entry
- Balanced traffic
- Normal vehicle mix

#### Evening Rush (16:00-19:00) 🔥
- Flow: ~1400 veh/h per entry  
- Heavy bias FROM urban center (eastbound)
- Maximum congestion period

#### Evening (19:00-22:00)
- Flow: ~400 veh/h per entry
- Moderate traffic

#### Night (22:00-24:00)
- Flow: ~80 veh/h per entry
- Truck ratio: 20% (freight)
- Light traffic

## Usage

### With SUMO GUI
```bash
sumo-gui -c simulation.sumocfg
```

### With SUMO-RL (for training)
```python
import sumo_rl

env = sumo_rl.parallel_env(
    net_file='scenarios/3x3/3x3_vietnamese/network.net.xml',
    route_file='scenarios/3x3/3x3_vietnamese/flows.rou.xml',
    use_gui=False,
    lateral_resolution=0.8,  # Enable sublane for motorcycles
    num_seconds=86400,  # 24 hours
    reward_fn='pressure',
    sumo_warnings=False
)
```

### Quick Test (1 hour morning rush)
```bash
sumo -c simulation.sumocfg --begin 21600 --end 25200
```

## Files
- `network.net.xml`: Road network
- `flows.rou.xml`: Traffic demand (24-hour)
- `vehicle_types.add.xml`: Vietnamese vehicle characteristics
- `simulation.sumocfg`: SUMO configuration

## Calibration Notes

Vehicle parameters calibrated for Vietnamese driving behavior:
- High sigma values (aggressive driving)
- Short tau (small headways)
- Lane-splitting enabled for motorcycles
- Speed factor >1.0 (speeding common)

## Citation

If you use this scenario in your research, please cite:
- SUMO: https://www.eclipse.org/sumo/
- Vietnamese traffic characteristics: Based on Hanoi/HCMC urban studies

Generated with Vietnamese Traffic Generator v1.0
