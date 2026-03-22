"""
config.py
---------
Central store for model hyperparameters and pipeline constants.

All values here were tuned via walk-forward grid search on 2024–2025 data.
To retune, run run_grid_search() from prophet_utils and update the relevant
dict below.
"""

# ── Regions & flows ───────────────────────────────────────────────────────────

REGIONS = ['South', 'West', 'North', 'East']

FLOWS = ['departure_handover', 'arrival_return', 'arrival_new_stock']

ARRIVALS   = ['arrival_new_stock', 'arrival_return', 'arrival_transfer_in']
DEPARTURES = ['departure_handover', 'departure_transfer_out', 'departure_remarketing']

# ── Overflow / capacity thresholds ───────────────────────────────────────────

OVERFLOW_THRESHOLD   = 0.9   # trigger planned transfer when projected stock > 90% capacity
EMERGENCY_THRESHOLD  = 0.99  # emergency dispatch when stock hits 99%

# ── Transfer cost constants (from constraints.csv) ───────────────────────────

TRUCK_BASE_FEE_EUR      = 45      # fixed dispatch fee per truck
TRUCK_PER_KM_EUR        = 0.55    # variable rate per km per truck
TRUCK_CAPACITY_CARS     = 8       # max cars per double-deck transporter
EMERGENCY_SURCHARGE     = 2.0     # cost multiplier for emergency dispatch
STANDARD_LEAD_TIME_DAYS = 2       # min booking lead time (planned)
EMERGENCY_LEAD_TIME_DAYS= 1       # min lead time for emergency dispatch
MAX_TRUCKS_PER_DAY      = 10      # network-wide daily truck cap
MAX_TRUCKS_PER_ROUTE    = 3       # per origin→destination per day

# ── Departure handover model ──────────────────────────────────────────────────

HO_REGRESSORS = ['new_subs_lag1', 'planned_ho_lag1']

HO_BEST = {
    # CV MAPE (walk-forward, min_train=18):
    # South=3.2%, West=8.9%, North=7.4%, East=7.0%
    'South': dict(changepoint_prior_scale=0.10, seasonality_prior_scale=20,
                  n_changepoints=5, seasonality_mode='additive'),
    'West':  dict(changepoint_prior_scale=0.05, seasonality_prior_scale=20,
                  n_changepoints=3, seasonality_mode='additive'),
    'North': dict(changepoint_prior_scale=0.05, seasonality_prior_scale=1,
                  n_changepoints=5, seasonality_mode='additive'),
    'East':  dict(changepoint_prior_scale=0.50, seasonality_prior_scale=20,
                  n_changepoints=5, seasonality_mode='multiplicative'),
}

# ── Arrival return model ──────────────────────────────────────────────────────

RET_REGRESSORS = ['returns_lag1', 'planned_ret_lag0', 'new_subs_lag1', 'backlog_lag1']

RET_BEST = {
    # CV MAPE: South=3.4%, West=8.3%, North=7.4%, East=2.8%
    'South': dict(changepoint_prior_scale=0.10, seasonality_prior_scale=1,
                  n_changepoints=5, seasonality_mode='multiplicative'),
    'West':  dict(changepoint_prior_scale=0.05, seasonality_prior_scale=1,
                  n_changepoints=3, seasonality_mode='multiplicative'),
    'North': dict(changepoint_prior_scale=0.05, seasonality_prior_scale=1,
                  n_changepoints=5, seasonality_mode='additive'),
    'East':  dict(changepoint_prior_scale=0.30, seasonality_prior_scale=1,
                  n_changepoints=5, seasonality_mode='additive'),
}

# ── Arrival new stock model ───────────────────────────────────────────────────

NS_REGRESSORS = ['planned_ho_lag1', 'new_subs_lag1']

NS_BEST = {
    # CV MAPE: South=6.0%, West=6.9%, North=10.0%, East=3.6%
    'South': dict(changepoint_prior_scale=0.05, seasonality_prior_scale=1,
                  n_changepoints=8, seasonality_mode='additive'),
    'West':  dict(changepoint_prior_scale=0.10, seasonality_prior_scale=10,
                  n_changepoints=5, seasonality_mode='additive'),
    'North': dict(changepoint_prior_scale=0.05, seasonality_prior_scale=1,
                  n_changepoints=3, seasonality_mode='additive'),
    'East':  dict(changepoint_prior_scale=0.05, seasonality_prior_scale=1,
                  n_changepoints=5, seasonality_mode='additive'),
}

# ── Forecast post-processing ─────────────────────────────────────────────────

# Fix 1: minimum weekly forecast as a fraction of that compound's monthly mean.
# Prevents Prophet from producing near-zero values for individual weeks (observed
# in September and December) while keeping the monthly total intact.
WEEKLY_FLOOR_RATIO = 0.20

# Fix 2: bias calibration window — number of prior years' same-month data used
# to compute per-compound calibration factors.  2 = use last 2 years of the same
# calendar month.  Set to 0 to disable calibration.
CALIBRATION_WINDOW_YEARS = 2

# Fix 3: recency window for compute_shares().  Only the last N months of events
# are used to estimate compound shares.  Prevents ramp-up data from anchoring
# new compounds (e.g. Leipzig) to unrepresentative early-months shares.
SHARES_RECENT_MONTHS = 12

# ── Compound weekly net-flow forecast params (logistic Prophet, cumulative net flow) ──
# Tuned via grid search (36 combos × 5 folds, horizon=1 MAPE) on 2024–2025 data.
# Keys: changepoint_prior_scale (cps), seasonality_prior_scale (sps), fourier_order (fo).
# _default applies to any compound not listed explicitly.

NET_FLOW_PROPHET_PARAMS = {
    '_default':         dict(changepoint_prior_scale=0.50, seasonality_prior_scale=1,  fourier_order=3),
    'CPD_MUNICH_01':    dict(changepoint_prior_scale=0.30, seasonality_prior_scale=5,  fourier_order=3),  # MAPE 1.6%
    'CPD_MUNICH_02':    dict(changepoint_prior_scale=0.50, seasonality_prior_scale=10, fourier_order=2),  # MAPE 4.2%
    'CPD_NUREMBERG_01': dict(changepoint_prior_scale=0.50, seasonality_prior_scale=1,  fourier_order=2),  # MAPE 17.7%
    'CPD_DUSSELDORF_01':dict(changepoint_prior_scale=0.50, seasonality_prior_scale=10, fourier_order=2),  # MAPE 10.9%
    'CPD_BREMEN_01':    dict(changepoint_prior_scale=0.50, seasonality_prior_scale=5,  fourier_order=2),  # MAPE 16.3%
    'CPD_HAMBURG_01':   dict(changepoint_prior_scale=0.30, seasonality_prior_scale=10, fourier_order=5),  # MAPE 6.8%
    'CPD_BERLIN_01':    dict(changepoint_prior_scale=0.50, seasonality_prior_scale=5,  fourier_order=5),  # MAPE 12.6%
    'CPD_DRESDEN_01':   dict(changepoint_prior_scale=0.05, seasonality_prior_scale=10, fourier_order=5),  # MAPE 17.9%
    'CPD_LEIPZIG_01':   dict(changepoint_prior_scale=0.15, seasonality_prior_scale=1,  fourier_order=2),  # MAPE 2.3%
}

# ── Compound share stability thresholds ──────────────────────────────────────

SHARE_COV_STABLE   = 0.20   # CoV < 0.20 → stable
SHARE_COV_MODERATE = 0.35   # CoV 0.20–0.35 → moderate; > 0.35 → unstable

# ── Prophet grid search default parameter grid ────────────────────────────────

DEFAULT_PARAM_GRID = {
    'changepoint_prior_scale': [0.05, 0.1, 0.3, 0.5],
    'seasonality_prior_scale': [1, 5, 10, 20],
    'n_changepoints':          [3, 5, 8],
    'seasonality_mode':        ['additive', 'multiplicative'],
}
