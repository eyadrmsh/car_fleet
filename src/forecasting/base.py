"""
base.py
-------
Abstract base class for all Prophet-based forecasters.

Every subclass must implement:
    _build_model()    — construct and configure a Prophet instance
    _prepare_train()  — slice and format training data
    _make_future()    — build the prediction DataFrame
    run_cv()          — walk-forward cross-validation
    forecast()        — fit on full history and predict
    run_grid_search() — hyperparameter search (classmethod)

Shared implementations provided here:
    _fit()            — silent Prophet fitting
    compute_metrics() — MAPE, MAE, bias, coverage, ci_excl_0 per horizon
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from prophet import Prophet


class ProphetForecaster(ABC):

    # ── Shared implementations ────────────────────────────────────────────────

    @staticmethod
    def _fit(model: Prophet, train_df: pd.DataFrame) -> Prophet:
        """Fit Prophet silently. train_df must have at least ds and y columns."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(train_df)
        return model

    @staticmethod
    def compute_metrics(cv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast quality per horizon.

        Returns
        -------
        DataFrame: horizon, n_folds, mape, mae, bias, coverage, ci_excl_0
        """
        rows = []
        for h in sorted(cv_df['horizon'].unique()):
            sub  = cv_df[cv_df['horizon'] == h]
            act  = sub['actual'].values
            pred = sub['predicted'].values
            rows.append({
                'horizon'   : h,
                'n_folds'   : len(sub),
                'mape'      : round(np.nanmean(
                                  np.abs((act - pred) / np.where(act == 0, np.nan, act)) * 100), 1),
                'mae'       : round(np.abs(act - pred).mean(), 1),
                'bias'      : round((pred - act).mean(), 1),
                'coverage'  : round(((sub['actual'] >= sub['lower_95']) &
                                     (sub['actual'] <= sub['upper_95'])).mean() * 100, 1),
                'ci_excl_0' : round(((sub['lower_95'] > 0) |
                                     (sub['upper_95'] < 0)).mean() * 100, 1),
            })
        return pd.DataFrame(rows)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> Prophet:
        """Construct and configure a Prophet instance for this forecaster."""

    @abstractmethod
    def _prepare_train(self, end_idx: int) -> pd.DataFrame:
        """Slice model_df up to end_idx and return a Prophet-ready DataFrame."""

    @abstractmethod
    def _make_future(self, *args, **kwargs) -> pd.DataFrame:
        """Build the single-row (or n-period) future DataFrame for prediction."""

    @abstractmethod
    def run_cv(self, **kwargs) -> pd.DataFrame:
        """Walk-forward cross-validation. Returns cv_df with actual vs predicted."""

    @abstractmethod
    def forecast(self, *args, **kwargs):
        """Fit on full history and predict future periods."""

    @classmethod
    @abstractmethod
    def run_grid_search(cls, *args, **kwargs):
        """Hyperparameter grid search. Returns best params per compound/region."""
