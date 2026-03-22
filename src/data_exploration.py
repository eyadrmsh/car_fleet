"""
data_exploration.py
-------------------
Analysis functions extracted from the data_exploration notebook.

Sections:
  1. Compound classification
  2. Event enrichment
  3. Monthly fullness
  4. Visualisation
  5. Demand rebalancing analysis
  6. East repair gap
  7. Repair bottleneck
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import compute_repair_share
from src.simulation.stock import (
    apply_repair_fifo,
    calibrate_initial_stock,
    build_stock_series,
)

# ── 1. Compound classification ────────────────────────────────────────────────

COMPOUND_ROLE_MAP = {
    'CPD_MUNICH_01':    'Overflow Source',
    'CPD_HAMBURG_01':   'Overflow Source',
    'CPD_FRANKFURT_01': 'Overflow Source',
    'CPD_COLOGNE_01':   'Overflow Source',
    'CPD_STUTTGART_01': 'Overflow Source',
    'CPD_LEIPZIG_01':   'Buffer',
    'CPD_DRESDEN_01':   'Buffer',
    'CPD_BREMEN_01':    'Buffer',
    'CPD_BERLIN_01':    'Repair Exporter',
    'CPD_MUNICH_02':    'Repair Exporter',
    'CPD_NUREMBERG_01': 'Repair Exporter',
    'CPD_DUSSELDORF_01':'Repair Exporter',
}


def compound_role_map() -> dict:
    """Return the hardcoded compound → role mapping."""
    return COMPOUND_ROLE_MAP.copy()


# ── 2. Event enrichment ───────────────────────────────────────────────────────

def build_true_events(car_events: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """
    Merge car_events with transfers and build a `true_event` column that
    annotates each event with its transfer reason or repair flag.
    """
    merged = car_events.merge(transfers, on='car_id', how='left')
    merged['event_date'] = pd.to_datetime(merged['event_date'])
    merged['event_month'] = merged['event_date'].dt.to_period('M')
    merged['true_event'] = np.where(
        merged['reason'].notna(),
        merged['event_type'] + '_' + merged['reason'],
        np.where(
            merged['needs_repair'] == True,
            merged['event_type'] + '_repair',
            merged['event_type'],
        ),
    )
    return merged


# ── 3. Monthly fullness ───────────────────────────────────────────────────────

def _get_repair_share_map(car_events: pd.DataFrame) -> tuple:
    """Return (share_map, global_share) from historical repair shares."""
    repair_shares = compute_repair_share(car_events)
    share_map = repair_shares.set_index('compound_id')['repair_share'].to_dict()
    global_share = float(repair_shares['repair_share'].mean())
    return share_map, global_share


def _compound_meta(compounds: pd.DataFrame) -> tuple:
    """Return (cap_map, rep_cap_map, capable_map) dicts keyed by compound_id."""
    cap_map = compounds.set_index('compound_id')['total_capacity'].to_dict()
    rep_cap_map = compounds.set_index('compound_id')['repair_capacity_per_week'].to_dict()
    capable_map = {
        r['compound_id']: str(r['repair_capable']).lower() in ['true', '1', 'yes']
        for _, r in compounds.iterrows()
    }
    return cap_map, rep_cap_map, capable_map


def _classify_repair_at_snap(rc, snap, rep_cap, is_capable, total, share_map, cpd, global_share):
    """Return (in_repair, waiting) for one compound at one snapshot date."""
    repair_cars_here = rc
    if is_capable:
        in_repair = int((
            (repair_cars_here['repair_start_date'] <= snap) &
            (repair_cars_here['repair_end_date'] > snap)
        ).sum())
        waiting = int((
            repair_cars_here['repair_end_date'].isna() |
            (repair_cars_here['repair_start_date'] > snap)
        ).sum())
        if rep_cap > 0:
            in_repair = min(in_repair, rep_cap)
    else:
        in_repair = 0
        waiting = int(round(total * share_map.get(cpd, global_share)))
    return in_repair, waiting


def _snap_compound_row(cpd, snap, cap, is_capable, rep_cap, stock_series, last_per_car, share_map, global_share):
    """Build the fullness dict for one compound at one snapshot date."""
    total = int(stock_series.loc[snap, cpd]) if (snap in stock_series.index and cpd in stock_series.columns) else 0
    rc = last_per_car[(last_per_car['compound_id'] == cpd) & (last_per_car['needs_repair'] == True)]
    in_repair, waiting = _classify_repair_at_snap(rc, snap, rep_cap, is_capable, total, share_map, cpd, global_share)
    available = max(0, total - in_repair - waiting)
    return {
        'month': snap.to_period('M'), 'compound_id': cpd,
        'total_capacity': cap, 'repair_capable': is_capable,
        'total': total, 'available': available,
        'in_repair': in_repair, 'waiting_repair': waiting,
        'utilisation': round(total / cap, 3) if cap > 0 else 0.0,
    }


def _snap_all_compounds(
    compounds: pd.DataFrame, snap: pd.Timestamp, stock_series: pd.DataFrame,
    last_per_car: pd.DataFrame, share_map: dict, global_share: float,
) -> list:
    """Build one row per compound for a single month-end snapshot."""
    cap_map, rep_cap_map, capable_map = _compound_meta(compounds)
    return [
        _snap_compound_row(
            cpd, snap,
            int(cap_map.get(cpd, 0)),
            capable_map.get(cpd, False),
            int(rep_cap_map.get(cpd, 0)),
            stock_series, last_per_car, share_map, global_share,
        )
        for cpd in compounds['compound_id']
    ]


def _prepare_repair_arrivals(car_events: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    """Run apply_repair_fifo and filter to arrival event types."""
    arrival_types = {'arrival_return', 'arrival_new_stock', 'arrival_transfer_in'}
    ev_repair = apply_repair_fifo(car_events, compounds)
    return ev_repair[ev_repair['event_type'].isin(arrival_types)].sort_values(
        ['car_id', 'event_date']
    )


def _build_stock_series_for_fullness(car_events, transfers, compounds, end):
    """Calibrate initial stock and build the daily stock series up to end."""
    initial_stock = calibrate_initial_stock(car_events, transfers, compounds)
    first_event = car_events['event_date'].min()
    return build_stock_series(car_events, compounds, initial_stock, start_date=first_event, end_date=end)


def build_monthly_fullness(car_events, transfers, compounds, start_date=None, end_date=None):
    """
    Monthly snapshot: total, available, in_repair, waiting_repair per compound.
    Calls apply_repair_fifo once (~30s). Pass ev externally to repair_queue_stats.
    """
    car_events = car_events.copy()
    car_events['event_date'] = pd.to_datetime(car_events['event_date'])
    end = pd.Timestamp(end_date) if end_date else car_events['event_date'].max()
    start = pd.Timestamp(start_date) if start_date else car_events['event_date'].min()
    month_ends = pd.date_range(start, end, freq='M')

    stock_series = _build_stock_series_for_fullness(car_events, transfers, compounds, end)
    repair_arrivals = _prepare_repair_arrivals(car_events, compounds)
    share_map, global_share = _get_repair_share_map(car_events)

    rows = []
    for snap in month_ends:
        last_per_car = (
            repair_arrivals[repair_arrivals['event_date'] <= snap]
            .groupby('car_id').last().reset_index()
        )
        rows.extend(_snap_all_compounds(compounds, snap, stock_series, last_per_car, share_map, global_share))
    return pd.DataFrame(rows)





# ── 3. Repair bottleneck ──────────────────────────────────────────────────────

def _repair_cars_with_wait(ev: pd.DataFrame) -> pd.DataFrame:
    """Filter ev to repair cars with a scheduled start and compute wait_days."""
    repair_cars = ev[
        (ev['needs_repair'] == True) & ev['repair_start_date'].notna()
    ].copy()
    repair_cars['wait_days'] = (
        repair_cars['repair_start_date'] - repair_cars['event_date']
    ).dt.days
    return repair_cars


def repair_queue_stats(ev: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    """
    Queue wait time per repair-capable compound.
    ev must be the output of apply_repair_fifo(car_events, compounds).
    """
    repair_cars = _repair_cars_with_wait(ev)
    return (
        repair_cars.groupby('compound_id')['wait_days']
        .agg(
            avg_wait='mean', median_wait='median', max_wait='max',
            pct_over_2w=lambda x: (x > 14).mean() * 100,
        )
        .round(1)
        .sort_values('avg_wait', ascending=False)
        .reset_index()
    )


def _throughput_row(cpd: str, cap: int, cpd_cars: pd.DataFrame) -> dict:
    """Compute throughput vs inflow metrics for one repair compound."""
    mean_days = cpd_cars['estimated_repair_days'].mean()
    throughput = cap * 7 / mean_days
    inflow = cpd_cars.groupby('week').size().mean()
    return {
        'compound_id': cpd, 'bays': cap,
        'mean_repair_days': round(mean_days, 1),
        'throughput_per_wk': round(throughput, 1),
        'inflow_per_wk': round(inflow, 1),
        'inflow/throughput': round(inflow / throughput, 2),
    }


def repair_throughput_vs_inflow(ev: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly throughput vs inflow per repair-capable compound.
    ev must be the output of apply_repair_fifo(car_events, compounds).
    """
    repair_cars = _repair_cars_with_wait(ev)
    repair_cars['week'] = pd.to_datetime(repair_cars['event_date']).dt.to_period('W')
    repair_caps_df = compounds[compounds['repair_capable'] == True][
        ['compound_id', 'repair_capacity_per_week']
    ]
    rows = []
    for _, r in repair_caps_df.iterrows():
        cpd_cars = repair_cars[repair_cars['compound_id'] == r['compound_id']]
        if not cpd_cars.empty:
            rows.append(_throughput_row(r['compound_id'], r['repair_capacity_per_week'], cpd_cars))
    return (
        pd.DataFrame(rows).sort_values('inflow/throughput', ascending=False).reset_index(drop=True)
    )


def repair_bay_utilisation(monthly_fullness: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    """Average and peak repair bay utilisation (in_repair / bays) per capable compound."""
    rep_cap_map = compounds.set_index('compound_id')['repair_capacity_per_week'].to_dict()
    capable = compounds[compounds['repair_capable'] == True]['compound_id'].tolist()
    util = monthly_fullness[monthly_fullness['compound_id'].isin(capable)].copy()
    util['bay_utilisation_pct'] = (
        util['in_repair'] / util['compound_id'].map(rep_cap_map) * 100
    ).round(1)
    return (
        util.groupby('compound_id')['bay_utilisation_pct']
        .agg(['mean', 'max']).round(1)
        .rename(columns={'mean': 'avg_bay_util_%', 'max': 'peak_bay_util_%'})
        .sort_values('avg_bay_util_%', ascending=False)
        .reset_index()
    )
