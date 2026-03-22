"""
monthly_flow.py
---------------
Regional monthly flow forecasting (handovers / returns / new stock).

Usage
-----
    from src.forecasting.monthly_flow import MonthlyFlowForecaster

    # CV for one region × flow
    cv_df = MonthlyFlowForecaster(region, flow, model_df, params, regressors).run_cv()

    # Forecast all regions × flows
    regional_fc = MonthlyFlowForecaster.run_monthly_forecast(
        events_df, demand_df, compounds_df, target_month)
"""

import logging
from itertools import product

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import ProphetForecaster
from src.config import (
    REGIONS,
    HO_REGRESSORS, HO_BEST,
    RET_REGRESSORS, RET_BEST,
    NS_REGRESSORS, NS_BEST,
    DEFAULT_PARAM_GRID,
)

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


# ── Forecaster class ──────────────────────────────────────────────────────────

class MonthlyFlowForecaster(ProphetForecaster):
    """
    Regional monthly flow forecasting (handovers / returns / new stock).

    Target variable : monthly event count per region
    Growth          : linear
    Frequency       : monthly
    Regressors      : external demand signals (e.g. new_subs_lag1, planned_ho_lag1)

    Parameters
    ----------
    region      : 'South' | 'West' | 'North' | 'East'
    flow        : 'departure_handover' | 'arrival_return' | 'arrival_new_stock'
    model_df    : DataFrame [ds, y, *regressors] — pre-built by _build_model_df_*
    params      : dict with changepoint_prior_scale, seasonality_prior_scale,
                  n_changepoints, seasonality_mode
    regressors  : list of regressor column names present in model_df
    changepoints: optional list of Timestamp-coercible dates to pin as changepoints
    """

    def __init__(self, region: str, flow: str, model_df: pd.DataFrame,
                 params: dict, regressors: list, changepoints: list = None):
        self.region       = region
        self.flow         = flow
        self.model_df     = model_df.copy()
        self.params       = params
        self.regressors   = regressors
        self.changepoints = changepoints

    # Internal helpers 

    def _build_model(self) -> Prophet:
        kwargs = dict(
            yearly_seasonality      = False,
            weekly_seasonality      = False,
            daily_seasonality       = False,
            seasonality_mode        = self.params['seasonality_mode'],
            changepoint_prior_scale = self.params['changepoint_prior_scale'],
            seasonality_prior_scale = self.params['seasonality_prior_scale'],
            n_changepoints          = self.params['n_changepoints'],
            interval_width          = 0.95,
        )
        if self.changepoints:
            kwargs['changepoints'] = [pd.Timestamp(c) for c in self.changepoints]
        m = Prophet(**kwargs)
        for reg in self.regressors:
            m.add_regressor(reg)
        return m

    def _prepare_train(self, end_idx: int) -> pd.DataFrame:
        return self.model_df.iloc[:end_idx][['ds', 'y'] + self.regressors].copy()

    def _make_future(self, test_row: dict) -> pd.DataFrame:
        return pd.DataFrame({'ds': [test_row['ds']]} |
                            {r: [test_row[r]] for r in self.regressors})

    # Instance interface 

    def run_cv(self, min_train: int = 18) -> pd.DataFrame:
        """
        Walk-forward CV: train on [:i], predict row [i] (one step ahead).

        Returns DataFrame: date, actual, predicted, error, abs_error, ape_pct,
                           lower_95, upper_95
        """
        n    = len(self.model_df)
        rows = []
        for i in range(min_train, n):
            train    = self._prepare_train(i)
            test_row = self.model_df.iloc[i].to_dict()
            model    = self._fit(self._build_model(), train)
            fc       = model.predict(self._make_future(test_row))
            y        = test_row['y']
            yhat     = fc['yhat'].values[0]
            rows.append(dict(
                date      = pd.Timestamp(test_row['ds']).strftime('%Y-%m'),
                actual    = y,
                predicted = round(yhat, 1),
                lower_95  = round(fc['yhat_lower'].values[0], 1),
                upper_95  = round(fc['yhat_upper'].values[0], 1),
                error     = round(y - yhat, 1),
                abs_error = round(abs(y - yhat), 1),
                ape_pct   = round(abs(y - yhat) / y * 100, 1) if y and y > 0 else np.nan,
            ))
        return pd.DataFrame(rows)

    def forecast(self, test_row: dict) -> dict:
        """
        Fit on full history, predict one month ahead.

        Returns dict: region, flow, date, predicted, lower_95, upper_95
        """
        train = self._prepare_train(len(self.model_df))
        model = self._fit(self._build_model(), train)
        fc    = model.predict(self._make_future(test_row))
        return dict(
            region    = self.region,
            flow      = self.flow,
            date      = pd.Timestamp(test_row['ds']).strftime('%Y-%m'),
            predicted = round(fc['yhat'].values[0], 1),
            lower_95  = round(fc['yhat_lower'].values[0], 1),
            upper_95  = round(fc['yhat_upper'].values[0], 1),
        )

    # Network-level classmethods 

    @classmethod
    def run_monthly_forecast(
        cls,
        events_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        compounds_df: pd.DataFrame,
        target_month: pd.Timestamp,
        ho_params: dict = None,
        ret_params: dict = None,
        ns_params: dict = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Retrain Prophet models on all data up to target_month - 1 and forecast
        target_month for all regions and flows.

        Returns
        -------
        DataFrame: month, region, flow, predicted, lower_95, upper_95
        """
        ho_params  = ho_params  or HO_BEST
        ret_params = ret_params or RET_BEST
        ns_params  = ns_params  or NS_BEST

        actuals = cls._build_actuals(events_df, compounds_df)
        demand  = demand_df.copy()
        demand['month'] = pd.to_datetime(demand['month']).dt.to_period('M')
        df = demand.merge(actuals, on=['month', 'region'], how='left')
        df = df[df['month'] <= pd.Period(target_month, 'M') - 1]

        rows = []
        for region in REGIONS:
            last = df[df['region'] == region].sort_values('month').iloc[-1]

            ho_row = dict(ds=target_month, y=np.nan,
                          new_subs_lag1=last['new_subscriptions'],
                          planned_ho_lag1=last['planned_handovers'])
            ho = cls(region, 'departure_handover',
                     cls._build_model_df_handover(region, df),
                     ho_params[region], HO_REGRESSORS).forecast(ho_row)
            rows.append(dict(month=target_month.strftime('%Y-%m'), region=region,
                             flow='departure_handover',
                             predicted=ho['predicted'],
                             lower_95=ho['lower_95'], upper_95=ho['upper_95']))

            mdf_ret = cls._build_model_df_returns(region, df)
            ret_row = dict(ds=target_month, y=np.nan,
                           returns_lag1=mdf_ret['y'].iloc[-1],
                           planned_ret_lag0=last['planned_returns'],
                           new_subs_lag1=last['new_subscriptions'],
                           backlog_lag1=last['backlog_orders'])
            ret = cls(region, 'arrival_return',
                      mdf_ret, ret_params[region], RET_REGRESSORS).forecast(ret_row)
            rows.append(dict(month=target_month.strftime('%Y-%m'), region=region,
                             flow='arrival_return',
                             predicted=ret['predicted'],
                             lower_95=ret['lower_95'], upper_95=ret['upper_95']))

            ns_row = dict(ds=target_month, y=np.nan,
                          planned_ho_lag1=last['planned_handovers'],
                          new_subs_lag1=last['new_subscriptions'])
            ns = cls(region, 'arrival_new_stock',
                     cls._build_model_df_new_stock(region, df),
                     ns_params[region], NS_REGRESSORS).forecast(ns_row)
            rows.append(dict(month=target_month.strftime('%Y-%m'), region=region,
                             flow='arrival_new_stock',
                             predicted=ns['predicted'],
                             lower_95=ns['lower_95'], upper_95=ns['upper_95']))

            if verbose:
                print(f'{region:8s}  HO={ho["predicted"]:6.0f}  '
                      f'RET={ret["predicted"]:6.0f}  NS={ns["predicted"]:6.0f}')

        return pd.DataFrame(rows)

    @classmethod
    def run_grid_search(
        cls,
        model_df: pd.DataFrame,
        regressors: list,
        param_grid: dict = None,
        loss: str = 'mape',
        sort_by: str = None,
        alpha: float = 2.0,
        min_train: int = 18,
        verbose: bool = True,
        extra_changepoints: list = None,
    ) -> pd.DataFrame:
        """
        Walk-forward grid search over Prophet hyperparameters for a monthly model.

        Returns
        -------
        DataFrame of all combos sorted by sort_by (best first)
        """
        param_grid = param_grid or DEFAULT_PARAM_GRID
        loss_to_col = {'mape': 'mape', 'mae': 'mae', 'rmse': 'rmse', 'asymmetric': 'asym_loss'}
        sort_col = sort_by if sort_by is not None else loss_to_col.get(loss, 'mape')
        if sort_col not in ('mape', 'mae', 'rmse', 'asym_loss'):
            raise ValueError(f"sort_by must be one of 'mape','mae','rmse','asym_loss'; got '{sort_col}'")

        combos = list(product(
            param_grid['changepoint_prior_scale'],
            param_grid['seasonality_prior_scale'],
            param_grid['n_changepoints'],
            param_grid['seasonality_mode'],
        ))
        n = len(model_df)
        if verbose:
            print(f'  {len(combos)} combos × {n - min_train} CV folds = '
                  f'{len(combos) * (n - min_train)} fits  |  sort_by={sort_col}  alpha={alpha}')

        results = []
        for i, (cps, sps, ncp, smode) in enumerate(combos):
            params = dict(changepoint_prior_scale=cps, seasonality_prior_scale=sps,
                          n_changepoints=ncp, seasonality_mode=smode)
            cv = cls(None, None, model_df, params, regressors,
                     changepoints=extra_changepoints).run_cv(min_train=min_train)
            results.append({
                **params,
                'mae':       cv['abs_error'].mean(),
                'mape':      cv['ape_pct'].mean(),
                'rmse':      np.sqrt((cv['error'] ** 2).mean()),
                'asym_loss': cls._asymmetric_loss(cv['actual'], cv['predicted'], alpha),
                'underpred': (cv['actual'] > cv['predicted']).mean(),
            })
            if verbose and ((i + 1) % 32 == 0 or (i + 1) == len(combos)):
                best = min(r[sort_col] for r in results)
                print(f'  {i+1}/{len(combos)} done — best {sort_col}: {best:.2f}')

        grid = pd.DataFrame(results).sort_values(sort_col).reset_index(drop=True)
        if verbose:
            print('\nTop 5:')
            print(grid.head(5)[['changepoint_prior_scale', 'seasonality_prior_scale',
                                 'n_changepoints', 'seasonality_mode',
                                 'mae', 'mape', 'rmse', 'asym_loss']]
                  .round(2).to_string())
        return grid

    # Model DataFrame builders (private staticmethods) 

    @staticmethod
    def _build_actuals(events_df: pd.DataFrame, compounds_df: pd.DataFrame) -> pd.DataFrame:
        ev = events_df.copy()
        ev['month'] = pd.to_datetime(ev['event_date']).dt.to_period('M')
        if 'region' not in ev.columns:
            ev = ev.merge(compounds_df[['compound_id', 'region']], on='compound_id', how='left')
        return ev.groupby(['month', 'region']).agg(
            real_handovers=('event_type', lambda x: (x == 'departure_handover').sum()),
            real_returns  =('event_type', lambda x: (x == 'arrival_return').sum()),
            real_new_stock=('event_type', lambda x: (x == 'arrival_new_stock').sum()),
        ).reset_index()

    @staticmethod
    def _build_model_df_handover(region: str, df: pd.DataFrame) -> pd.DataFrame:
        sub = df[df['region'] == region][
            ['month', 'new_subscriptions', 'planned_handovers', 'real_handovers']
        ].copy().sort_values('month').reset_index(drop=True)
        sub['new_subs_lag1']   = sub['new_subscriptions'].shift(1)
        sub['planned_ho_lag1'] = sub['planned_handovers'].shift(1)
        sub.rename(columns={'month': 'ds', 'real_handovers': 'y'}, inplace=True)
        sub['ds'] = sub['ds'].dt.to_timestamp()
        return sub[['ds', 'y'] + HO_REGRESSORS].dropna().reset_index(drop=True)

    @staticmethod
    def _build_model_df_returns(region: str, df: pd.DataFrame) -> pd.DataFrame:
        sub = df[df['region'] == region][
            ['month', 'new_subscriptions', 'planned_returns', 'backlog_orders', 'real_returns']
        ].copy().sort_values('month').reset_index(drop=True)
        sub['returns_lag1']     = sub['real_returns'].shift(1)
        sub['planned_ret_lag0'] = sub['planned_returns']
        sub['new_subs_lag1']    = sub['new_subscriptions'].shift(1)
        sub['backlog_lag1']     = sub['backlog_orders'].shift(1)
        sub.rename(columns={'month': 'ds', 'real_returns': 'y'}, inplace=True)
        sub['ds'] = sub['ds'].dt.to_timestamp()
        return sub[['ds', 'y'] + RET_REGRESSORS].dropna().reset_index(drop=True)

    @staticmethod
    def _build_model_df_new_stock(region: str, df: pd.DataFrame) -> pd.DataFrame:
        sub = df[df['region'] == region][
            ['month', 'new_subscriptions', 'planned_handovers', 'real_new_stock']
        ].copy().sort_values('month').reset_index(drop=True)
        sub['planned_ho_lag1'] = sub['planned_handovers'].shift(1)
        sub['new_subs_lag1']   = sub['new_subscriptions'].shift(1)
        sub.rename(columns={'month': 'ds', 'real_new_stock': 'y'}, inplace=True)
        sub['ds'] = sub['ds'].dt.to_timestamp()
        return sub[['ds', 'y'] + NS_REGRESSORS].dropna().reset_index(drop=True)

    @staticmethod
    def _asymmetric_loss(actual, predicted, alpha: float = 2.0) -> float:
        """Pinball loss: under-prediction penalised alpha× more than over-prediction."""
        errors = np.array(actual) - np.array(predicted)
        return np.where(errors > 0, alpha * errors, -errors).mean()


# ── Module-level alias (backward compat) ──────────────────────────────────────
run_monthly_forecast = MonthlyFlowForecaster.run_monthly_forecast
