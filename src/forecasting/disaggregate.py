"""
disaggregate.py
---------------
Turn regional monthly forecasts into compound-level weekly forecasts.

Pipeline
--------
1. compute_shares()         learn compound shares within region from historical events
2. apply_compound_shares()           regional monthly totals → compound monthly totals
3. build_weekly_weights()   learn within-month weekly patterns per compound from history
4. apply_weekly_weights()   compound monthly totals → compound weekly forecasts

Usage
-----
    from src.forecasting.disaggregate import (
        compute_shares, apply_compound_shares,
        build_weekly_weights, apply_weekly_weights,
    )

    # Step 1: compute shares from training data (pass only data up to forecast cutoff)
    shares = compute_shares(events_df)

    # Step 2: regional → compound monthly
    compound_monthly = apply_compound_shares(monthly_forecast_df, shares)

    # Step 3: learn weekly patterns from same month in prior years
    weekly_weights = build_weekly_weights(events_df, target_month=1, ref_years=[2024, 2025])

    # Step 4: compound monthly → compound weekly
    weekly = apply_weekly_weights(compound_monthly, weekly_weights, pd.Timestamp('2026-01-01'))
    # Returns DataFrame: [week_start, region, compound_id, event_type, weekly_forecast]
"""

import numpy as np
import pandas as pd

from src.config import (
    FLOWS, SHARE_COV_STABLE, SHARE_COV_MODERATE,
    WEEKLY_FLOOR_RATIO, CALIBRATION_WINDOW_YEARS, SHARES_RECENT_MONTHS,
)
from src.preprocessing import add_date_parts


# Helpers

def _month_week_map(year: int, target_month: int) -> pd.DataFrame:
    """Return DataFrame [week_start, week_num] for all weeks overlapping target_month in year."""
    month_start = pd.Timestamp(year=year, month=target_month, day=1)
    month_end   = month_start + pd.offsets.MonthEnd(0)
    days        = pd.date_range(month_start, month_end, freq='D')
    week_starts = sorted(pd.Series(days).dt.to_period('W').dt.start_time.unique())
    return pd.DataFrame({
        'week_start': week_starts,
        'week_num':   range(1, len(week_starts) + 1),
    })


def _normalize_weights(series: pd.Series) -> pd.Series:
    """Normalize a Series to sum to 1; falls back to uniform if total is zero."""
    total = series.sum()
    if total > 0:
        return series / total
    return pd.Series([1 / len(series)] * len(series), index=series.index)


def _filter_target_month(events_df: pd.DataFrame, target_month: int) -> pd.DataFrame:
    """Filter events to target month, parse dates, add year/month/week_start columns."""
    ev       = add_date_parts(events_df[events_df['event_type'].isin(FLOWS)])
    month_ev = ev[ev['month'] == target_month].copy()
    return month_ev


def _assign_week_nums(month_ev: pd.DataFrame, target_month: int) -> tuple:
    """
    Attach week_num to each event row and return (annotated_df, week_map_df).
    week_map_df has columns [year, week_start, week_num].
    """
    all_years   = month_ev['year'].unique()
    week_map_df = pd.concat(
        [_month_week_map(year, target_month).assign(year=year) for year in all_years],
        ignore_index=True,
    )
    return month_ev.merge(week_map_df, on=['year', 'week_start'], how='left'), week_map_df


def _weekly_shares(month_ev: pd.DataFrame, week_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each compound × flow × week_num's share of its annual monthly total.
    """
    all_years = month_ev['year'].unique()
    all_cpds  = month_ev['compound_id'].unique()
    max_weeks = week_map_df['week_num'].max()

    weekly = (
        month_ev.groupby(['year', 'compound_id', 'event_type', 'week_num'])
        .size().reset_index(name='n')
    )
    monthly_totals = (
        weekly.groupby(['year', 'compound_id', 'event_type'])['n']
        .sum().reset_index(name='month_total')
    )

    full_idx = pd.MultiIndex.from_product(
        [all_years, all_cpds, FLOWS, range(1, max_weeks + 1)],
        names=['year', 'compound_id', 'event_type', 'week_num'],
    )
    weekly = (weekly.set_index(['year', 'compound_id', 'event_type', 'week_num'])
              .reindex(full_idx, fill_value=0).reset_index())
    weekly = weekly.merge(monthly_totals, on=['year', 'compound_id', 'event_type'])
    weekly['share'] = np.where(
        weekly['month_total'] > 0,
        weekly['n'] / weekly['month_total'],
        0.0,
    )
    return weekly[['year', 'compound_id', 'event_type', 'week_num', 'share']]


def _avg_and_normalise(weekly_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Average weekly shares across years per compound × flow × week_num
    """
    avg = (
        weekly_shares.groupby(['compound_id', 'event_type', 'week_num'])['share']
        .mean().reset_index(name='avg_share')
    )
    rows = []
    for (cpd, flow), grp in avg.groupby(['compound_id', 'event_type']):
        grp = grp.sort_values('week_num').copy()
        w   = _normalize_weights(grp['avg_share'])
        for wn, wval in zip(grp['week_num'].values, w.values):
            rows.append(dict(
                compound_id = cpd,
                event_type  = flow,
                week_num    = int(wn),
                weight      = round(float(wval), 6),
            ))
    return pd.DataFrame(rows)


# Functions

def compute_shares(events_df: pd.DataFrame,
                   flows: list = None,
                   recent_months: int = None) -> pd.DataFrame:
    """
    Derive compound-within-region shares from historical event data.

    recent_months : if set, only the last N months of events are used.
        Prevents ramp-up data (e.g. Leipzig's first months) from anchoring
        shares to unrepresentative early volumes.  Default: use all history.
        SHARES_RECENT_MONTHS from config is the recommended value.
    """
    flows = flows or FLOWS
    ev = events_df[events_df['event_type'].isin(flows)].copy()
    ev['event_date'] = pd.to_datetime(ev['event_date'])

    if recent_months:
        cutoff = ev['event_date'].max() - pd.DateOffset(months=recent_months)
        ev = ev[ev['event_date'] > cutoff]

    ev['month'] = ev['event_date'].dt.to_period('M')

    monthly = (
        ev.groupby(['month', 'region', 'compound_id', 'event_type'])
        .size().reset_index(name='n')
    )
    region_totals = (
        monthly.groupby(['month', 'region', 'event_type'])['n']
        .sum().reset_index(name='region_total')
    )
    monthly = monthly.merge(region_totals, on=['month', 'region', 'event_type'])
    monthly['share'] = monthly['n'] / monthly['region_total']

    stats = (
        monthly.groupby(['region', 'compound_id', 'event_type'])['share']
        .agg(mean_share='mean', std_share='std')
        .reset_index()
    )

    region_sum = (
        stats.groupby(['region', 'event_type'])['mean_share']
        .sum().reset_index(name='share_sum')
    )
    stats = stats.merge(region_sum, on=['region', 'event_type'])
    stats['mean_share'] = (stats['mean_share'] / stats['share_sum']).round(4)
    stats = stats.drop(columns='share_sum')

    stats['cov'] = (stats['std_share'] / stats['mean_share']).fillna(0).round(3)
    stats['label'] = pd.cut(
        stats['cov'],
        bins=[-np.inf, SHARE_COV_STABLE, SHARE_COV_MODERATE, np.inf],
        labels=['stable', 'moderate', 'unstable'],
    ).astype(str)

    return stats[['event_type', 'region', 'compound_id', 'mean_share', 'cov', 'label']]



def apply_compound_shares(regional_forecast: pd.DataFrame,
                          shares_df: pd.DataFrame,
                          forecast_col: str = 'predicted') -> pd.DataFrame:
    """
    Multiply regional forecast totals by compound shares.

    Parameters
    ----------
    forecast_col : which Prophet output column to use as the regional total.
                   'predicted'  — point estimate (default)
                   'upper_95'   — upper 95% CI, conservative for service planning
                   'lower_95'   — lower 95% CI, for cost/capacity lower-bound
    """
    if forecast_col not in regional_forecast.columns:
        raise ValueError(f"forecast_col='{forecast_col}' not in regional_forecast columns: "
                         f"{list(regional_forecast.columns)}")
    fc = regional_forecast.rename(columns={'flow': 'event_type', forecast_col: 'region_forecast'})
    merged = shares_df.merge(
        fc[['region', 'event_type', 'region_forecast']],
        on=['region', 'event_type'],
    )
    merged['compound_forecast'] = (merged['region_forecast'] * merged['mean_share']).round(1)
    return merged[['region', 'compound_id', 'event_type', 'compound_forecast']]


def build_weekly_weights(events_df: pd.DataFrame,
                         target_month: int = 1) -> pd.DataFrame:
    """
    Learn per-compound within-month weekly weight vectors from historical data.

    New compounds that have no events in target_month (e.g. opened after the
    last occurrence of that month) receive uniform weights so they are not
    silently dropped by apply_weekly_weights.
    """
    month_ev              = _filter_target_month(events_df, target_month)
    month_ev, week_map_df = _assign_week_nums(month_ev, target_month)
    shares                = _weekly_shares(month_ev, week_map_df)
    weights               = _avg_and_normalise(shares)

    # Fallback for new compounds with no target-month history ──────────────
    # Any compound that appears in events_df (for any month) but has no
    # target-month events gets uniform weights — equal split across weeks.
    all_cpds    = set(events_df[events_df['event_type'].isin(FLOWS)]['compound_id'].unique())
    covered     = set(weights['compound_id'].unique())
    new_cpds    = all_cpds - covered
    n_weeks     = int(week_map_df['week_num'].max())

    if new_cpds:
        fallback = pd.DataFrame([
            dict(compound_id=cpd, event_type=flow,
                 week_num=wn, weight=round(1.0 / n_weeks, 6))
            for cpd  in new_cpds
            for flow in FLOWS
            for wn   in range(1, n_weeks + 1)
        ])
        weights = pd.concat([weights, fallback], ignore_index=True)

    check = weights.groupby(['compound_id', 'event_type'])['weight'].sum().round(4)
    assert (check == 1.0).all(), f'Weight vectors do not sum to 1.0: {check[check != 1.0]}'
    return weights


def apply_weekly_weights(compound_monthly: pd.DataFrame,
                         weights_df: pd.DataFrame,
                         target_month: pd.Timestamp,
                         floor_ratio: float = WEEKLY_FLOOR_RATIO) -> pd.DataFrame:
    """
    Distribute compound monthly forecasts to weekly level.
    """
    week_map = _month_week_map(target_month.year, target_month.month)
    n_weeks  = len(week_map)

    merged = compound_monthly.merge(
        weights_df[['compound_id', 'event_type', 'week_num', 'weight']],
        on=['compound_id', 'event_type'],
    )
    merged = merged.merge(week_map, on='week_num')
    merged['weekly_forecast'] = merged['compound_forecast'] * merged['weight']

    if floor_ratio > 0:
        grp_key = ['compound_id', 'event_type']
        monthly_mean_per_week = merged.groupby(grp_key)['compound_forecast'].first() / n_weeks
        floor_map = (monthly_mean_per_week * floor_ratio).to_dict()

        def _apply_floor(grp):
            key   = (grp['compound_id'].iloc[0], grp['event_type'].iloc[0])
            floor = floor_map.get(key, 0.0)
            vals  = grp['weekly_forecast'].values.copy()
            below = vals < floor
            if below.any() and vals.sum() > 0:
                vals[below] = floor
                # Re-scale so monthly total is preserved
                total   = grp['compound_forecast'].iloc[0]
                current = vals.sum()
                if current > 0:
                    vals = vals * (total / current)
            grp = grp.copy()
            grp['weekly_forecast'] = vals
            return grp

        merged = merged.groupby(grp_key, group_keys=False).apply(_apply_floor)

    merged['weekly_forecast'] = merged['weekly_forecast'].round(1)

    return (merged[['week_start', 'region', 'compound_id', 'event_type', 'weekly_forecast']]
            .sort_values(['event_type', 'compound_id', 'week_start'])
            .reset_index(drop=True))


def apply_bias_calibration(weekly_fc: pd.DataFrame,
                           events_df: pd.DataFrame,
                           target_month: pd.Timestamp,
                           window_years: int = CALIBRATION_WINDOW_YEARS,
                           clip_range: tuple = (0.7, 1.5)) -> pd.DataFrame:
    """
    For each compound × flow, compare the current forecast monthly total against
    the actual monthly totals for the same calendar month in the last `window_years`
    years of training data.  The ratio (actual_mean / forecast) is applied as a
    multiplicative correction, clipped to `clip_range` to avoid over-correction.

    Example: if Berlin's forecast says 100 handovers for January but the last
    2 Januaries averaged 115, the calibration factor is 1.15 and the forecast is
    scaled up accordingly.

    Parameters
    ----------
    weekly_fc    : output of apply_weekly_weights()
    events_df    : training events (strictly before target_month — no leakage)
    target_month : the month being forecast
    window_years : how many prior same-month years to average over
    clip_range   : (min, max) multiplier bounds to prevent wild corrections
    """
    if window_years <= 0:
        return weekly_fc

    cal_month  = target_month.month
    ev = events_df[events_df['event_type'].isin(FLOWS)].copy()
    ev['event_date'] = pd.to_datetime(ev['event_date'])
    ev['year']       = ev['event_date'].dt.year
    ev['month_num']  = ev['event_date'].dt.month

    max_year  = target_month.year - 1
    min_year  = max_year - window_years + 1
    same_month = ev[(ev['month_num'] == cal_month) & ev['year'].between(min_year, max_year)]

    if same_month.empty:
        return weekly_fc

    actual = (
        same_month.groupby(['compound_id', 'event_type', 'year'])
        .size().reset_index(name='actual_count')
    )
    actual_mean = (
        actual.groupby(['compound_id', 'event_type'])['actual_count']
        .mean().reset_index(name='actual_mean')
    )

    fc_monthly = (
        weekly_fc.groupby(['compound_id', 'event_type'])['weekly_forecast']
        .sum().reset_index(name='fc_total')
    )

    calib = actual_mean.merge(fc_monthly, on=['compound_id', 'event_type'])
    calib['factor'] = (
        calib['actual_mean'] / calib['fc_total'].clip(lower=1)
    ).clip(*clip_range).round(4)

    factor_map = calib.set_index(['compound_id', 'event_type'])['factor'].to_dict()

    result = weekly_fc.copy()
    result['_factor'] = result.apply(
        lambda r: factor_map.get((r['compound_id'], r['event_type']), 1.0), axis=1
    )
    result['weekly_forecast'] = (result['weekly_forecast'] * result['_factor']).round(1)
    return result.drop(columns='_factor')
