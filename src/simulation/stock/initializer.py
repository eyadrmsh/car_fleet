"""
initializer.py
--------------
Pre-simulation stock setup: anchor initial stock from historical data
and build a daily stock series for each compound.
"""

import numpy as np
import pandas as pd

from src.config import ARRIVALS, DEPARTURES


def build_net_flow(compound_id: str, car_events: pd.DataFrame) -> pd.Series:
    """Cumulative daily net car count for a compound (+1 arrival, -1 departure)."""
    ev = car_events[
        (car_events['compound_id'] == compound_id) &
        (car_events['event_type'].isin(ARRIVALS + DEPARTURES))
    ].copy()
    ev['event_date'] = pd.to_datetime(ev['event_date'])
    ev['sign'] = ev['event_type'].apply(lambda x: 1 if x in ARRIVALS else -1)
    return ev.groupby('event_date')['sign'].sum().cumsum().rename('net_flow')


def get_initial_stock(compound_id: str,
                      car_events: pd.DataFrame,
                      transfers: pd.DataFrame,
                      compounds_df: pd.DataFrame) -> float:
    """
    Estimate stock before the event log starts.
    Anchors on the first overflow transfer: at that moment the compound was
    at ~80% capacity, so initial = 0.8 × (capacity − net_count_at_overflow_date).
    Returns 0 if no overflow history exists.
    """
    cap_map  = compounds_df.set_index('compound_id')['total_capacity'].to_dict()
    capacity = cap_map.get(compound_id)
    if capacity is None:
        return 0.0

    src_col = 'from_compound_id' if 'from_compound_id' in transfers.columns else 'from_compound'
    trf = transfers.copy()
    trf['transfer_date'] = pd.to_datetime(trf['transfer_date'])
    overflow = trf[
        (trf[src_col] == compound_id) & (trf['reason'] == 'capacity_overflow')
    ].sort_values('transfer_date')

    if overflow.empty:
        return 0.0

    first_overflow_date = overflow['transfer_date'].iloc[0]
    net_flow = build_net_flow(compound_id, car_events)
    past = net_flow[net_flow.index <= first_overflow_date]
    count_at_overflow = float(past.iloc[-1]) if not past.empty else 0.0
    return max(0.0, 0.93 * (capacity - count_at_overflow))


def calibrate_initial_stock(car_events: pd.DataFrame,
                             transfers: pd.DataFrame,
                             compounds_df: pd.DataFrame) -> dict:
    """Return {compound_id: int} for every compound using overflow-anchored estimation."""
    return {
        cpd: int(round(get_initial_stock(cpd, car_events, transfers, compounds_df)))
        for cpd in compounds_df['compound_id']
    }


def build_stock_series(car_events: pd.DataFrame,
                       compounds_df: pd.DataFrame,
                       initial_stock: dict,
                       start_date,
                       end_date) -> pd.DataFrame:
    """
    Daily stock count for each compound from start_date to end_date.
    Returns a DataFrame indexed by date, one column per compound_id.
    """
    date_idx = pd.date_range(start_date, end_date, freq='D')
    result = {}
    for cpd in compounds_df['compound_id']:
        net_flow = build_net_flow(cpd, car_events)
        init = float(initial_stock.get(cpd, 0))
        nf = net_flow.reindex(date_idx).ffill().fillna(0)
        result[cpd] = (init + nf).clip(lower=0)
    return pd.DataFrame(result, index=date_idx)
