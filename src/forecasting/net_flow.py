"""
net_flow.py
-----------
Compound-level cumulative net flow forecasting.

Usage
-----
    from src.forecasting.net_flow import NetFlowForecaster, build_weekly_net

    weekly_net   = build_weekly_net(car_events, compounds)
    changepoints = NetFlowForecaster.infer_changepoints_gradient(weekly_net)

    # CV for all compounds
    cv_results = NetFlowForecaster.run_network_cv(
        weekly_net, compounds, NET_FLOW_PROPHET_PARAMS, changepoints)

    # Forecast for all compounds
    forecast_df = NetFlowForecaster.run_network_forecast(
        weekly_net, compounds, NET_FLOW_PROPHET_PARAMS, forecast_weeks, changepoints)

    # Convert to simulation input
    weekly_deltas = NetFlowForecaster.to_weekly_net_deltas(forecast_df, weekly_net)
    # {week: {compound_id: delta}} — plug into run_weekly_rules rolling stock
"""

import warnings
from itertools import product as _product

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA

from .base import ProphetForecaster


# ── Data preparation ──────────────────────────────────────────────────────────

def build_weekly_net(car_events: pd.DataFrame,
                     compounds: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate car_events into weekly cumulative net flow per compound.

    net_flow = cumulative sum of (+1 arrival, -1 departure) per compound,
    starting from the first event in the data.

    Returns
    -------
    DataFrame with columns: compound_id, region, week_start, net_flow, rolling_net_flow
    """
    ev = car_events.copy()
    ev['event_date'] = pd.to_datetime(ev['event_date'])

    if 'region' not in ev.columns:
        ev = ev.merge(compounds[['compound_id', 'region']], on='compound_id', how='left')

    ev['week_start'] = ev['event_date'].dt.to_period('W').dt.start_time
    ev['signed']     = ev['event_type'].map(lambda t: +1 if t.startswith('arrival') else -1)

    weekly = (
        ev.groupby(['compound_id', 'region', 'week_start'])['signed']
        .sum().reset_index(name='net_flow')
        .sort_values(['compound_id', 'week_start'])
    )

    weekly['net_flow'] = weekly.groupby('compound_id')['net_flow'].cumsum()
    weekly['rolling_net_flow'] = (
        weekly.groupby('compound_id')['net_flow']
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

    return weekly.reset_index(drop=True)


# ── Forecaster class ──────────────────────────────────────────────────────────

class NetFlowForecaster(ProphetForecaster):
    """
    Compound-level cumulative net flow forecasting.

    Target variable : cumulative weekly net flow (arrivals − departures)
    Growth          : logistic  (cap = compound capacity, floor = 0)
    Frequency       : weekly
    Changepoints    : inferred from gradient zero-crossings (passed in at construction)

    Parameters
    ----------
    compound_id  : e.g. 'CPD_MUNICH_01'
    weekly_net   : output of build_weekly_net() — must contain compound_id, week_start, net_flow
    params       : dict with changepoint_prior_scale, seasonality_prior_scale, fourier_order
    cap          : compound total_capacity (logistic upper bound)
    changepoints : list of date strings e.g. ['2024-03-04', '2024-09-02']
    """

    def __init__(self, compound_id: str, weekly_net: pd.DataFrame,
                 params: dict, cap: float, changepoints: list):
        self.compound_id  = compound_id
        self.params       = params
        self.cap          = cap
        self.changepoints = changepoints
        self._data        = (weekly_net[weekly_net['compound_id'] == compound_id]
                             .sort_values('week_start').reset_index(drop=True))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_model(self, train_max_ds: pd.Timestamp = None) -> Prophet:
        cp = [c for c in self.changepoints
              if train_max_ds is None or pd.Timestamp(c) < train_max_ds]

        kwargs = dict(
            growth                  = 'logistic',
            yearly_seasonality      = True,
            weekly_seasonality      = True,
            daily_seasonality       = False,
            seasonality_mode        = 'additive',
            changepoint_prior_scale = self.params['changepoint_prior_scale'],
            seasonality_prior_scale = self.params['seasonality_prior_scale'],
            interval_width          = 0.95,
        )
        kwargs['changepoints' if cp else 'n_changepoints'] = (
            [pd.Timestamp(c) for c in cp] if cp else 0
        )
        m = Prophet(**kwargs)
        m.add_seasonality(name='yearly', period=52.18,
                          fourier_order=self.params['fourier_order'])
        return m

    def _prepare_train(self, end_idx: int) -> pd.DataFrame:
        train = self._data.iloc[:end_idx][['week_start', 'net_flow']].rename(
            columns={'week_start': 'ds', 'net_flow': 'y'})
        train['cap']   = self.cap
        train['floor'] = 0.0
        return train

    def _make_future(self, model: Prophet, n_periods: int) -> pd.DataFrame:
        future = model.make_future_dataframe(periods=n_periods, freq='W-MON')
        future['cap']   = self.cap
        future['floor'] = 0.0
        return future

    def _ar_correction(self, model: Prophet, train: pd.DataFrame,
                       n_steps: int) -> tuple:
        """AR(p) correction on in-sample residuals. Returns (corrections, ar_order)."""
        future_in = self._make_future(model, 0)
        fitted    = model.predict(future_in).set_index('ds')['yhat']
        residuals = (train.set_index('ds')['y'] - fitted).dropna().values
        threshold = 2.0 / np.sqrt(len(residuals))
        acf_vals  = acf(residuals, nlags=min(8, len(residuals) // 4), fft=False)
        sig_lags  = [l for l in range(1, len(acf_vals)) if abs(acf_vals[l]) > threshold]
        if not sig_lags:
            return np.zeros(n_steps), 0
        try:
            ar_order    = min(max(sig_lags), 2)
            corrections = ARIMA(residuals, order=(ar_order, 0, 0)).fit().forecast(n_steps)
            return corrections.values, ar_order
        except Exception:
            return np.zeros(n_steps), 0

    # Instance interface

    def run_cv(self, min_train: int = 52, n_horizons: int = 4) -> pd.DataFrame:
        """
        Walk-forward CV: train on [:i], predict weeks [i … i+n_horizons].

        Returns DataFrame: fold, horizon, week, actual, predicted, lower_95, upper_95
        """
        n    = len(self._data)
        rows = []
        for i in range(min_train, n - n_horizons + 1):
            train = self._prepare_train(i)
            model = self._fit(self._build_model(train['ds'].max()), train)
            fc    = model.predict(self._make_future(model, n_horizons)).iloc[i:]

            for h_idx, (_, fc_row) in enumerate(fc.iterrows()):
                act_idx = i + h_idx
                if act_idx >= n:
                    break
                rows.append(dict(
                    fold      = i,
                    horizon   = h_idx + 1,
                    week      = self._data.iloc[act_idx]['week_start'],
                    actual    = self._data.iloc[act_idx]['net_flow'],
                    predicted = fc_row['yhat'],
                    lower_95  = fc_row['yhat_lower'],
                    upper_95  = fc_row['yhat_upper'],
                ))
        return pd.DataFrame(rows)

    def forecast(self, forecast_weeks: list) -> pd.DataFrame:
        """
        Fit on full history, apply AR correction, predict forecast_weeks.

        Returns DataFrame: compound_id, compound, week, horizon,
                           predicted, lower_95, upper_95, ar_order
        """
        train               = self._prepare_train(len(self._data))
        model               = self._fit(self._build_model(), train)
        corrections, ar_ord = self._ar_correction(model, train, len(forecast_weeks))

        future = self._make_future(model, len(forecast_weeks))
        fc     = model.predict(future)
        fc_out = fc[fc['ds'].isin(forecast_weeks)].reset_index(drop=True)

        label = self.compound_id.replace('CPD_', '').replace('_01', '')
        rows  = []
        for h_idx, (_, fc_row) in enumerate(fc_out.iterrows()):
            corr = corrections[h_idx] if h_idx < len(corrections) else 0
            rows.append(dict(
                compound_id = self.compound_id,
                compound    = label,
                week        = fc_row['ds'].date(),
                horizon     = h_idx + 1,
                predicted   = round(fc_row['yhat']       + corr, 1),
                lower_95    = round(fc_row['yhat_lower'] + corr, 1),
                upper_95    = round(fc_row['yhat_upper'] + corr, 1),
                ar_order    = ar_ord,
            ))
        return pd.DataFrame(rows)

    # Network-level classmethods

    @staticmethod
    def infer_changepoints_gradient(
        weekly_net: pd.DataFrame,
        flow_col: str = 'rolling_net_flow',
        min_prominence: float = 2.0,
        min_gap_weeks: int = 3,
    ) -> dict:
        """
        Find changepoints as zero-crossings of the rolling net flow gradient.

        Returns
        -------
        dict {compound_id: [date_str, ...]}
        """
        result = {}
        for cpd, grp in weekly_net.groupby('compound_id'):
            grp   = grp.sort_values('week_start').reset_index(drop=True)
            vals  = grp[flow_col].fillna(0).values
            dates = grp['week_start'].values
            grad  = np.gradient(vals)

            candidates = [
                pd.Timestamp(dates[i])
                for i in range(1, len(grad))
                if (np.sign(grad[i]) != np.sign(grad[i - 1])
                    and grad[i - 1] != 0
                    and abs(vals[i]) >= min_prominence)
            ]
            changepoints, last = [], None
            for cp in candidates:
                if last is None or (cp - last).days >= min_gap_weeks * 7:
                    changepoints.append(cp.strftime('%Y-%m-%d'))
                    last = cp
            if changepoints:
                result[cpd] = changepoints
        return result

    @classmethod
    def run_network_cv(
        cls,
        weekly_net: pd.DataFrame,
        compounds_df: pd.DataFrame,
        params: dict,
        changepoints: dict = None,
        min_train: int = 52,
        n_horizons: int = 4,
    ) -> dict:
        """
        Run walk-forward CV for every compound.

        Returns
        -------
        dict {compound_id: cv_df}
        """
        changepoints = changepoints or {}
        cap_map      = compounds_df.set_index('compound_id')['total_capacity'].to_dict()
        results      = {}
        for cpd in compounds_df['compound_id']:
            p   = params.get(cpd, params.get('_default', {}))
            cap = float(cap_map.get(cpd, 9999))
            results[cpd] = cls(cpd, weekly_net, p, cap,
                               changepoints.get(cpd, [])).run_cv(
                               min_train=min_train, n_horizons=n_horizons)
        return results

    @classmethod
    def run_network_forecast(
        cls,
        weekly_net: pd.DataFrame,
        compounds_df: pd.DataFrame,
        params: dict,
        forecast_weeks: list,
        changepoints: dict = None,
    ) -> pd.DataFrame:
        """
        Forecast all compounds and combine into one DataFrame.

        Returns
        -------
        DataFrame: compound_id, compound, region, week, horizon,
                   predicted, lower_95, upper_95, ar_order
        """
        changepoints = changepoints or {}
        cap_map      = compounds_df.set_index('compound_id')['total_capacity'].to_dict()
        region_map   = compounds_df.set_index('compound_id')['region'].to_dict()
        rows         = []
        for cpd in compounds_df['compound_id']:
            p     = params.get(cpd, params.get('_default', {}))
            cap   = float(cap_map.get(cpd, 9999))
            fc_df = cls(cpd, weekly_net, p, cap,
                        changepoints.get(cpd, [])).forecast(forecast_weeks)
            fc_df['region'] = region_map.get(cpd, '')
            rows.append(fc_df)
        return pd.concat(rows, ignore_index=True)

    @staticmethod
    def to_weekly_net_deltas(
        forecast_df: pd.DataFrame,
        weekly_net: pd.DataFrame,
    ) -> dict:
        """
        Convert cumulative net flow forecast to per-week stock deltas for simulation.

        NetFlowForecaster predicts cumulative net flow. The simulation's run_weekly_rules
        needs the weekly change: delta = cumulative[this_week] - cumulative[prev_week].
        For horizon=1, prev is the last known historical value.

        Returns
        -------
        dict {week (Timestamp): {compound_id: int}}
        """
        last_known = (
            weekly_net.sort_values('week_start')
            .groupby('compound_id')['net_flow']
            .last()
            .to_dict()
        )

        result = {}
        for week in sorted(forecast_df['week'].unique()):
            week_ts   = pd.Timestamp(week)
            week_rows = forecast_df[forecast_df['week'] == week]
            deltas    = {}
            for _, row in week_rows.iterrows():
                cpd = row['compound_id']
                h   = int(row['horizon'])
                prev_rows = forecast_df[
                    (forecast_df['compound_id'] == cpd) & (forecast_df['horizon'] == h - 1)
                ]
                prev_val = (prev_rows['predicted'].values[0]
                            if h > 1 and not prev_rows.empty
                            else last_known.get(cpd, 0))
                deltas[cpd] = int(round(row['predicted'] - prev_val))
            result[week_ts] = deltas
        return result

    @classmethod
    def run_grid_search(
        cls,
        weekly_net: pd.DataFrame,
        compounds_df: pd.DataFrame,
        changepoints: dict = None,
        min_train: int = 52,
        n_gs_folds: int = 5,
        param_grid: dict = None,
        verbose: bool = True,
    ) -> dict:
        """
        Walk-forward grid search over Prophet hyperparameters for every compound.

        Uses only the last n_gs_folds folds and horizon=1 for speed.

        Returns
        -------
        dict {compound_id: best_params_dict}
        """
        changepoints = changepoints or {}
        cap_map      = compounds_df.set_index('compound_id')['total_capacity'].to_dict()

        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.05, 0.15, 0.30, 0.50],
                'seasonality_prior_scale': [1, 5, 10],
                'fourier_order'          : [2, 3, 5],
            }

        combos = list(_product(
            param_grid['changepoint_prior_scale'],
            param_grid['seasonality_prior_scale'],
            param_grid['fourier_order'],
        ))
        if verbose:
            n_cpd = len(compounds_df)
            print(f'{len(combos)} combos × {n_gs_folds} folds × {n_cpd} compounds '
                  f'= {len(combos) * n_gs_folds * n_cpd:,} Prophet fits')

        best_params = {}
        for cpd in compounds_df['compound_id']:
            cap     = float(cap_map.get(cpd, 9999))
            cp      = changepoints.get(cpd, [])
            sub     = (weekly_net[weekly_net['compound_id'] == cpd]
                       .sort_values('week_start').reset_index(drop=True))
            start_i = max(min_train, len(sub) - n_gs_folds)

            results = []
            for cps, sps, fo in combos:
                p    = dict(changepoint_prior_scale=cps,
                            seasonality_prior_scale=int(sps),
                            fourier_order=int(fo))
                mape = cls._gs_mape_one(sub, cap, cp, p, start_i)
                results.append({**p, 'mape': mape})

            df_res = pd.DataFrame(results).sort_values('mape').reset_index(drop=True)
            best   = df_res.iloc[0]
            best_params[cpd] = dict(
                changepoint_prior_scale = best['changepoint_prior_scale'],
                seasonality_prior_scale = int(best['seasonality_prior_scale']),
                fourier_order           = int(best['fourier_order']),
            )
            if verbose:
                label = cpd.replace('CPD_', '').replace('_01', '')
                print(f"  {label:18s}  MAPE={best['mape']:.1f}%  "
                      f"cps={best['changepoint_prior_scale']}  "
                      f"sps={int(best['seasonality_prior_scale'])}  "
                      f"fo={int(best['fourier_order'])}")

        return best_params

    @staticmethod
    def _gs_mape_one(sub: pd.DataFrame, cap: float, changepoints: list,
                     params: dict, start_i: int) -> float:
        """One-step-ahead MAPE for one compound × one param combo (grid search only)."""
        actuals, preds = [], []
        n = len(sub)
        for i in range(start_i, n):
            cp = [c for c in changepoints if pd.Timestamp(c) < sub.iloc[i]['week_start']]
            kwargs = dict(
                growth='logistic',
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                interval_width=0.95,
            )
            kwargs['changepoints' if cp else 'n_changepoints'] = (
                [pd.Timestamp(c) for c in cp] if cp else 0
            )
            m = Prophet(**kwargs)
            m.add_seasonality(name='yearly', period=52.18, fourier_order=params['fourier_order'])

            train = sub.iloc[:i][['week_start', 'net_flow']].rename(
                columns={'week_start': 'ds', 'net_flow': 'y'})
            train['cap'] = cap
            train['floor'] = 0.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m.fit(train)
                future = pd.DataFrame({'ds': [sub.iloc[i]['week_start']], 'cap': [cap], 'floor': [0.0]})
                preds.append(m.predict(future)['yhat'].values[0])
                actuals.append(sub.iloc[i]['net_flow'])
            except Exception:
                continue

        if not actuals:
            return np.nan
        act  = np.array(actuals)
        pred = np.array(preds)
        return float(np.nanmean(np.abs((act - pred) / np.where(act == 0, np.nan, act)) * 100))


# ── Module-level aliases (backward compat) ────────────────────────────────────
infer_changepoints_gradient = NetFlowForecaster.infer_changepoints_gradient
run_network_cv              = NetFlowForecaster.run_network_cv
run_network_forecast        = NetFlowForecaster.run_network_forecast
to_weekly_net_deltas        = NetFlowForecaster.to_weekly_net_deltas
