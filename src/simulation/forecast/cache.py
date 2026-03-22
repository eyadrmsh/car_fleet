"""
forecast_cache.py
-----------------
ForecastCache — caches monthly compound weekly forecasts and repair shares,
re-training automatically when the simulation enters a new calendar month.
"""

from typing import Optional

import pandas as pd

from src.preprocessing import compute_repair_share
from src.forecasting import MonthlyFlowForecaster
from src.forecasting import (
    compute_shares, apply_compound_shares,
    build_weekly_weights, apply_weekly_weights,
    apply_bias_calibration,
)
from src.config import SHARES_RECENT_MONTHS, CALIBRATION_WINDOW_YEARS


class ForecastCache:
    """
    Lazy monthly forecaster.  Call refresh_if_needed() at the top of each
    week in the simulation loop; the cache re-trains only when the month rolls over.

    Public attributes after first refresh
    ----------------------------------------
    weekly_fc      : DataFrame [week_start, region, compound_id, event_type, weekly_forecast]
    repair_shares  : output of compute_repair_share()
    mean_days_map  : {compound_id: mean_repair_days}

    Parameters
    ----------
    forecast_col : which Prophet output column to use as the regional total.
                   'predicted' (default) — point estimate
                   'upper_95'            — conservative, better for service planning
                   'lower_95'            — optimistic lower bound
    """

    def __init__(self, forecast_col: str = 'predicted'):
        self._month:          Optional[pd.Timestamp] = None
        self._forecast_col:   str = forecast_col
        self.weekly_fc:       Optional[pd.DataFrame] = None
        self.repair_shares:   Optional[pd.DataFrame] = None
        self.mean_days_map:   dict = {}
        self.remarketing_rates: dict = {}

    # ── Internal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _month_ts(week_monday: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(year=week_monday.year, month=week_monday.month, day=1)

    def _retrain(self,
                 month_ts:     pd.Timestamp,
                 car_events:   pd.DataFrame,
                 demand_df:    pd.DataFrame,
                 compounds_df: pd.DataFrame) -> None:
        ev_train = car_events[car_events['event_date'] < month_ts].copy()

        self.repair_shares = compute_repair_share(ev_train)
        self.mean_days_map = (
            self.repair_shares.set_index('compound_id')['mean_repair_days'].to_dict()
        )


        rm = ev_train[ev_train['event_type'] == 'departure_remarketing'].copy()
        if not rm.empty:
            rm['event_date'] = pd.to_datetime(rm['event_date'])
            rm['week'] = rm['event_date'].dt.to_period('W')
            rm_weekly = rm.groupby(['compound_id', 'week']).size().reset_index(name='n')
            self.remarketing_rates = (
                rm_weekly.groupby('compound_id')['n'].mean().round(2).to_dict()
            )
        else:
            self.remarketing_rates = {}

        regional_fc      = MonthlyFlowForecaster.run_monthly_forecast(ev_train, demand_df, compounds_df, month_ts)

        shares           = compute_shares(ev_train, recent_months=SHARES_RECENT_MONTHS)
        compound_monthly = apply_compound_shares(regional_fc, shares,
                                                 forecast_col=self._forecast_col)
        weights          = build_weekly_weights(ev_train, target_month=month_ts.month)
        weekly_raw       = apply_weekly_weights(compound_monthly, weights, month_ts)
        self.weekly_fc   = apply_bias_calibration(weekly_raw, ev_train, month_ts,
                                                  window_years=CALIBRATION_WINDOW_YEARS)



    def refresh_if_needed(self,
                          week_monday:  pd.Timestamp,
                          car_events:   pd.DataFrame,
                          demand_df:    pd.DataFrame,
                          compounds_df: pd.DataFrame) -> bool:
        """
        Re-train all models when week_monday falls in a new calendar month.
        Returns True if a refresh occurred, False if cache was already current.
        """
        month_ts = self._month_ts(week_monday)
        if month_ts == self._month:
            return False

        self._month = month_ts
        self._retrain(month_ts, car_events, demand_df, compounds_df)
        return True

    def _slice_fc(self, fc_df: pd.DataFrame, week_monday: pd.Timestamp) -> pd.DataFrame:
        """Slice a forecast DataFrame for the given week with fallback."""
        wk = fc_df[fc_df['week_start'] == week_monday]
        if not wk.empty:
            return wk
        past = fc_df[fc_df['week_start'] <= week_monday]
        return (
            past.sort_values('week_start')
                .groupby(['compound_id', 'event_type'])
                .last()
                .reset_index()
        )

    def get_week_fc(self, week_monday: pd.Timestamp) -> pd.DataFrame:
        """
        Slice weekly_fc for the given week.
        Falls back to the most-recent earlier week if an exact match is not found
        (handles cases where week_monday spans two months and the forecast only
        covers weeks within the current month).
        """
        return self._slice_fc(self.weekly_fc, week_monday)

