# FINN Compound Network — Project Guide

## What this project is

A compound logistics forecasting and simulation system for FINN's car fleet network in Germany. The goal is to predict stock levels at each compound, schedule repair work, and optimise truck dispatches — before overflow situations happen rather than reacting to them.

**Core problem**: Fleet grew 40% YoY but compound capacity hasn't kept pace. Each unplanned overflow transfer costs 2× the standard rate. The logistics partner caps the network at 10 trucks/day.

---

## Compound network

12 compounds across 4 regions:

| Region | Compounds |
|--------|-----------|
| South  | Munich 01 (1200, repair), Munich 02 (350), Stuttgart (500, repair), Nuremberg (300) |
| West   | Cologne (600, repair), Düsseldorf (400), Frankfurt (700, repair) |
| North  | Hamburg (900, repair), Bremen (250) |
| East   | Berlin (800), Leipzig (300, opened Jul 2024), Dresden (200) |

Repair-capable compounds have simultaneous bay capacity (e.g. Munich 01 = 60 bays). Non-repair compounds hold cars waiting for dispatch to a repair compound.

---

### Coding rules

1. Extract logic into a function when any of these are true:

It's used more than once anywhere in the codebase
It has a clear name that makes the code more readable
It's more than ~5 lines of non-trivial logic
It could reasonably be tested in isolation

2. Functions do one thing only — if you use "and" to describe it, split it
load_and_summarise → two functions: load_csv() and summarise_dataframe()

3. Constants never live inside functions — they go in a config or constants file
No magic numbers like > 0.8 inside logic. Use STRESS_THRESHOLD_PCT from config.

4. Prompts are never built with f-strings scattered across the pipeline
Each prompt is assembled by a dedicated builder function in its own file.

5. No function longer than 30 lines — if it's longer, break it up
run_pipeline() should read like a table of contents, not implementation:

## Source files (`src/`)

### `config.py`
All constants. Key values:
- `OVERFLOW_THRESHOLD = 0.9` — trigger planned transfer at 90% utilisation
- `TRUCK_CAPACITY_CARS = 8`, `MAX_TRUCKS_PER_DAY = 10`, `MAX_TRUCKS_PER_ROUTE = 3`
- `STANDARD_LEAD_TIME_DAYS = 2` — minimum booking lead time for planned dispatches
- `EMERGENCY_SURCHARGE = 2.0` — emergency dispatch costs 2× standard
- `TRUCK_BASE_FEE_EUR = 45`, `TRUCK_PER_KM_EUR = 0.55`
- Prophet hyperparameters `HO_BEST`, `RET_BEST`, `NS_BEST` — tuned via walk-forward CV, do not change without re-running grid search



**Key design decisions**:
- Cars at non-repair-capable compounds always show as `waiting_repair` — they cannot be fixed there. This is correct.
- `repair_capacity_per_week` = simultaneous bays, not weekly throughput.

---

## Data files

| File | Description |
|------|-------------|
| `car_events.csv` | Main event log. Always apply `flag_repair_cars` then `impute_repair_estimate` before using. |
| `transfers.csv` | Historical transfer records. `cost_eur` is **per car** (truck cost ÷ cars on truck). Summing all rows for a truck equals formula cost. |
| `compounds.csv` | Compound metadata — capacity, location, lat/lon, repair capability |
| `demand_forecast_inputs.csv` | Monthly regional demand signals (new_subscriptions, planned_handovers, etc.) |
| `constraints.csv` | Logistics constraints (truck rates, caps) |

---

## Dispatch plan constraints (all enforced in `consolidate_and_schedule`)

| Constraint | Value | Rule |
|---|---|---|
| Network daily cap | 10 trucks/day | No day exceeds this across all routes |
| Per-route daily cap | 3 trucks/route/day | Routes needing more trucks split across consecutive days |
| Lead time | 2 days | Earliest dispatch = week_monday + `STANDARD_LEAD_TIME_DAYS` |
| Truck fill | 100% | Only full truckloads dispatched; partial loads deferred to next week |

Trucks are **contracted carriers** (not owned fleet). Lead time = booking lead time, not physical positioning time. The model does not track truck location between trips.

---


## Working with notebooks

Always execute the notebook and report actual output — never describe what it *would* produce. 


