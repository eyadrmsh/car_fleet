# Compound Network — Logistics Forecasting & Simulation

Predicts stock levels at each compound, schedules repair work, and optimises truck dispatches proactively — before overflow happens rather than reacting to it.

**Core problem**: Fleet grew 40% YoY but compound capacity hasn't kept pace. Each unplanned overflow transfer costs 2× the standard rate. The logistics partner caps the network at 10 trucks/day.

---

## Compound network

| Region | Compound | Capacity | Repair |
|--------|----------|----------|--------|
| South | Munich 01 | 1,200 | ✓  |
| South | Munich 02 | 350 | |
| South | Stuttgart | 500 | ✓ |
| South | Nuremberg | 300 | |
| West | Cologne | 600 | ✓ |
| West | Düsseldorf | 400 | |
| West | Frankfurt | 700 | ✓ |
| North | Hamburg | 900 | ✓ |
| North | Bremen | 250 | |
| East | Berlin | 800 | |
| East | Leipzig | 300 | (opened Jul 2024) |
| East | Dresden | 200 | |

---

## Repository structure

```
FINN/
├── data/                    # Input CSVs (car_events, transfers, compounds, demand, constraints)
├── notebooks/               # Analysis and simulation notebooks
├── src/
│   ├── config.py            # All constants and Prophet hyperparameters — never hardcode elsewhere
│   ├── preprocessing.py     # Car-level data preparation (flag repairs, impute durations, join region)
│   ├── data_exploration.py  # Reusable analysis functions used by notebooks
│   ├── forecasting/         # Regional monthly → compound weekly Prophet forecast pipeline
│   │   ├── base.py          # Abstract ProphetForecaster base class
│   │   ├── monthly_flow.py  # Regional monthly forecasting (handover / return / new stock)
│   │   ├── net_flow.py      # Compound-level cumulative net flow forecasting
│   │   └── disaggregate.py  # Disaggregate regional monthly totals to compound × week
│   └── simulation/          # Week-by-week compound network simulation
│       ├── state.py         # SimulationState dataclass
│       ├── runner.py        # HorizonSimulationRunner — main weekly orchestration loop
│       ├── dispatch/        # Rules for car transfers launching
│       │   ├── emergency.py # Fires immediately at ≥ 99% utilisation (2× surcharge)
│       │   ├── projector.py # Projects stock N weeks ahead using the forecast
│       │   ├── overflow.py  # Proactive overflow dispatch for compounds forecast to breach 90%
│       │   ├── repair.py    # RepairBatcher — consolidates repair cars into full truckloads
│       │   └── scheduler.py # Packs moves into a truck schedule; enforces daily caps + lead times
│       ├── forecast/        # Functionality to use prophet forecast pipiline in simulation
│       │   ├── cache.py     # ForecastCache — lazy monthly Prophet retraining
│       │   └── repair_tracker.py  # Identifies which repair spots are occupied and which cars are queued in compound, based on repair schedule
│       ├── reporting/
│       │   └── logger.py    # Tracks cost and handover shortfall per week
│       └── stock/
│           ├── initializer.py  # Calibrate initial stock from historical events
│           ├── repair_fifo.py  # Creates repair schedule (repair_start/end_date per car)
│           ├── status.py       # Compound snapshot — utilisation, repair queue, waiting
│           └── updater.py      # Apply weekly net flow + truck moves to stock

```

---

## Notebooks

| Notebook | What it does |
|----------|-------------|
| `data_exploration.ipynb` | Compound role classification, monthly fullness, transfer breakdown by reason, repair capacity bottlenecks |
| `monthly_flows_forecast.ipynb` | Walk-forward CV for all three regional flows; forecast vs actual 2025 |
| `compound_net_flow_weekly_forecast.ipynb` | Compound-level weekly net flow forecast using `NetFlowForecaster`; 4-week ahead for Jan 2026 |
| `compound_share_analysis.ipynb` | CoV stability analysis per compound per flow; identifies unstable compounds (e.g. Leipzig ramp-up) |
| `repair_rate_analysis.ipynb` | Repair share by segment and brand; quarterly stability; compound-level rates |
| `overflow_reduction_2025.ipynb` | Full 2025 simulation — actual vs baseline vs optimised cost comparison. Baseline saves 30%, optimised saves 46% vs actual spend |

---

## Running a simulation

```python
import pandas as pd
from src.preprocessing import flag_repair_cars, impute_repair_estimate, add_region_to_car_events
from src.simulation import SimulationState, HorizonSimulationRunner
from src.simulation.stock import apply_repair_fifo, calibrate_initial_stock, build_stock_series

# 1. Load data
car_events = pd.read_csv('data/car_events.csv', parse_dates=['event_date'])
transfers  = pd.read_csv('data/transfers.csv',  parse_dates=['transfer_date', 'scheduled_date'])
compounds  = pd.read_csv('data/compounds.csv',  parse_dates=['opened_date'])
demand     = pd.read_csv('data/demand_forecast_inputs.csv', parse_dates=['month'])

# 2. Preprocess — always in this order
car_events = flag_repair_cars(car_events, transfers)
car_events = impute_repair_estimate(car_events)
car_events = add_region_to_car_events(car_events, compounds)

# 3. Build initial stock at simulation start
SIM_START = pd.Timestamp('2025-01-06')
ev_pre    = car_events[car_events['event_date'] < SIM_START].copy()
ev_sched  = apply_repair_fifo(ev_pre, compounds)

initial_stock = calibrate_initial_stock(car_events, transfers, compounds)
stock_series  = build_stock_series(car_events, compounds, initial_stock, '2024-12-01', SIM_START)
stock_week0   = stock_series.iloc[-1].to_dict()

state = SimulationState(stock=stock_week0, repair_queue={}, repair_load={}, week=SIM_START)

# 4. Run
runner = HorizonSimulationRunner(
    car_events     = car_events,
    demand         = demand,
    compounds      = compounds,
    dist_matrix    = dist_matrix,   # haversine dict keyed by (compound_id, compound_id)
    ev_train_sched = ev_sched,
)
runner.run(state, start_date=SIM_START, end_date=pd.Timestamp('2025-12-28'))

# 5. Results
print(runner.summary())
cost_df     = runner.cost_df()      # trucks, cars, cost_eur per week × reason
handover_df = runner.handover_df()  # demand, available, shortfall per week × compound
```

**Key options on `HorizonSimulationRunner`:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `horizon_weeks` | 4 | How many weeks ahead to project when making overflow decisions |
| `urgency_window` | 2 | Only dispatch proactively if breach is forecast within this many weeks |
| `use_repair_batching` | `False` | `True` = `RepairBatcher` consolidates repair cars across senders |
| `actual_transfers` | `None` | If set, these transfers are appended as-is each week (bypasses scheduling) |
