"""
preprocessing.py
----------------
Car-level preprocessing for repair status and repair duration.

Functions
---------
flag_repair_cars         — enforce needs_repair=True from transfer routing + arrival returns
impute_repair_estimate   — fill missing estimated_repair_days by random draw or median
add_region_to_car_events — merge region column from compounds onto car_events
add_date_parts           — add year, month, week_start columns derived from event_date
compute_repair_share     — retrospective share of arrival_returns needing repair + mean duration

"""

import numpy as np
import pandas as pd


def flag_repair_cars(car_events: pd.DataFrame,
                     transfers: pd.DataFrame) -> pd.DataFrame:
    car_events = car_events.merge(transfers[['car_id', 'reason']], on='car_id', how='left')
    car_events['needs_repair'] = np.where(car_events['reason'] == 'repair_routing', True, car_events['needs_repair'])
    return car_events.drop(['reason'], axis=1)


def impute_repair_estimate(car_events: pd.DataFrame,
                           mode: str = 'median',
                           seed: int = None) -> pd.DataFrame:
    
    car_events = car_events.copy()
    if seed is not None:
        np.random.seed(seed)

    mask = (car_events['needs_repair'] == True) & (car_events['estimated_repair_days'].isna())

    if mask.sum() == 0:
        print('All cars to repair have time estimate')
        return car_events

    existing = car_events.loc[car_events['needs_repair'] == True, 'estimated_repair_days'].dropna()

    if mode == 'median':
        car_events.loc[mask, 'estimated_repair_days'] = existing.median()
    elif mode == 'random':
        car_events.loc[mask, 'estimated_repair_days'] = np.random.choice(
            existing.values, size=mask.sum(), replace=True)

    return car_events


def add_region_to_car_events(car_events: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    return car_events.merge(compounds[['compound_id', 'region']], on='compound_id', how='left')


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, and week_start columns derived from event_date."""
    df = df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['year']       = df['event_date'].dt.year
    df['month']      = df['event_date'].dt.month
    df['week_start'] = df['event_date'].dt.to_period('W').dt.start_time
    return df


def compute_repair_share(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrospectively compute the share of arrival_return events that need repair,
    and the mean repair duration, from historical data.
    """
    returns = events_df[events_df['event_type'] == 'arrival_return'].copy()

    stats = returns.groupby('compound_id').agg(
            n_returns      = ('event_id', 'count'),
            n_repair       = ('needs_repair', 'sum'),
            mean_repair_days = ('estimated_repair_days', 'mean'),
        ).reset_index()

    stats['repair_share'] = (stats['n_repair'] / stats['n_returns']).round(4)
    stats['mean_repair_days'] = stats['mean_repair_days'].round(1)
    return stats