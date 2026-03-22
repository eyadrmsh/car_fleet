"""
runner.py
---------
HorizonSimulationRunner — week-by-week orchestration loop.

This class is a thin coordinator: it holds references to all components,
calls them in order each week, and delegates results to SimulationLogger.
No dispatch or stock logic lives here.

Usage
-----
    from src.simulation import SimulationState, HorizonSimulationRunner

    state  = SimulationState(stock=..., repair_queue=..., repair_load=..., week=start)
    runner = HorizonSimulationRunner(car_events, demand, compounds, dist_matrix, ev_train_sched)
    runner.run(state, start_date='2025-01-06', end_date='2025-12-28')

    print(runner.summary())
    cost_df     = runner.cost_df()
    handover_df = runner.handover_df()
"""

import pandas as pd

from src.simulation.state import SimulationState
from src.simulation.forecast.cache import ForecastCache
from src.simulation.forecast.repair_tracker import RepairStateTracker
from src.simulation.dispatch.emergency import EmergencyDispatcher
from src.simulation.dispatch.projector import HorizonProjector
from src.simulation.dispatch.overflow import OverflowCalendar, RouteBatcher
from src.simulation.dispatch.repair import RepairBatcher
from src.simulation.dispatch.scheduler import consolidate_and_schedule
from src.simulation.stock.updater import update_stock
from src.simulation.reporting.logger import SimulationLogger
from src.config import EMERGENCY_LEAD_TIME_DAYS


_ORGANIC_TYPES = {
    'arrival_return', 'arrival_new_stock',
    'departure_handover', 'departure_remarketing',
}


def _apply_moves(stock: dict, actions: list) -> dict:
    """Return a copy of stock with the given moves applied (no floor)."""
    s = dict(stock)
    for a in actions:
        s[a['from_compound']] = max(0, s.get(a['from_compound'], 0) - a['cars'])
        s[a['to_compound']]   = s.get(a['to_compound'], 0) + a['cars']
    return s


class HorizonSimulationRunner:
    """
    Parameters
    ----------
    car_events          : full car event log
    demand              : monthly demand forecast inputs
    compounds           : compound metadata
    dist_matrix         : {(from_id, to_id): km}
    ev_train_sched      : FIFO-scheduled pre-sim car events (repair bays assigned)
    horizon_weeks       : weeks ahead to project for overflow detection (default 4)
    urgency_window      : dispatch only if overflow is within this many weeks (default 2)
    forecast_col        : Prophet output column to use ('predicted', 'upper_95', 'lower_95')
    use_repair_batching : include cross-compound repair dispatch (default True)
    actual_transfers    : optional transfers.csv DataFrame — when provided, non-overflow
                          transfers for each week are included in the truck schedule.
                          Use this in the baseline scenario to reflect actual historical
                          repair routing, demand rebalancing, and remarketing moves.
    """

    def __init__(
        self,
        car_events:          pd.DataFrame,
        demand:              pd.DataFrame,
        compounds:           pd.DataFrame,
        dist_matrix:         dict,
        ev_train_sched:      pd.DataFrame,
        horizon_weeks:       int  = 4,
        urgency_window:      int  = 2,
        forecast_col:        str  = 'predicted',
        use_repair_batching: bool = True,
        actual_transfers:    pd.DataFrame = None,
    ):
        self._car_events         = car_events
        self._demand             = demand
        self._compounds          = compounds
        self._dist_matrix        = dist_matrix
        self.horizon_weeks       = horizon_weeks
        self.urgency_window      = urgency_window
        self.use_repair_batching = use_repair_batching

        self._forecast            = ForecastCache(forecast_col=forecast_col)
        self._repair_tracker      = RepairStateTracker(ev_train_sched, compounds)
        self._emergency           = EmergencyDispatcher(compounds, dist_matrix)
        self.logger               = SimulationLogger()
        self._actual_transfers    = self._prepare_transfers(actual_transfers)

    # ── Actual-transfers helpers ─────────────────────────────────────────────

    @staticmethod
    def _prepare_transfers(transfers: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and drop overflow rows once at init time."""
        if transfers is None:
            return None
        df = transfers.copy()
        df['transfer_date'] = pd.to_datetime(df['transfer_date'])
        # Normalise compound column names (transfers.csv uses _id suffix)
        if 'from_compound_id' in df.columns:
            df = df.rename(columns={
                'from_compound_id': 'from_compound',
                'to_compound_id':   'to_compound',
            })
        return df[df['reason'] != 'capacity_overflow'].reset_index(drop=True)

    def _actual_transfer_actions(self,
                                  week_monday: pd.Timestamp,
                                  week_end:    pd.Timestamp) -> list:
        """
        Convert actual historical transfers for this week into action dicts
        compatible with consolidate_and_schedule.

        One action per unique truck: cars = cars_per_truck, cost = sum of
        per-car cost_eur (= total truck cost), trucks = 1.
        """
        if self._actual_transfers is None:
            return []
        week_trf = self._actual_transfers[
            self._actual_transfers['transfer_date'].between(week_monday, week_end)
        ]
        if week_trf.empty:
            return []
        return (
            week_trf.groupby('truck_id')
            .agg(
                from_compound = ('from_compound', 'first'),
                to_compound   = ('to_compound',   'first'),
                cars          = ('cars_per_truck', 'first'),
                distance_km   = ('distance_km',   'first'),
                reason        = ('reason',         'first'),
                cost_eur      = ('cost_eur',        'sum'),   # per-car × n_cars = truck cost
            )
            .assign(trucks=1)
            .reset_index(drop=True)
            .to_dict('records')
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_multi_fc(self, week_monday: pd.Timestamp) -> pd.DataFrame:
        return pd.concat([
            self._forecast.get_week_fc(
                week_monday + pd.Timedelta(weeks=i)
            ).assign(week_start=week_monday + pd.Timedelta(weeks=i))
            for i in range(self.horizon_weeks)
        ], ignore_index=True)

    def _get_repair_actions(self, repair_load, repair_queue, projected,
                            multi_fc, week_monday, stock) -> list:
        if not self.use_repair_batching:
            return []
        avail = {
            cpd: max(0, stock.get(cpd, 0)
                     - repair_load.get(cpd, 0)
                     - repair_queue.get(cpd, 0))
            for cpd in stock
        }
        ho_fc = (
            multi_fc[
                (multi_fc['week_start'] == week_monday) &
                (multi_fc['event_type'] == 'departure_handover')
            ]
            .groupby('compound_id')['weekly_forecast'].sum()
            .to_dict()
        )
        return RepairBatcher().batch(
            repair_queue    = repair_queue,
            projected       = projected,
            repair_load     = repair_load,
            compounds_df    = self._compounds,
            dist_matrix     = self._dist_matrix,
            available_stock = avail,
            ho_weekly_fc    = ho_fc,
        )

    def _build_schedule(self, emergency_actions, regular_actions,
                        week_monday) -> pd.DataFrame:
        em_sched = (
            consolidate_and_schedule(
                emergency_actions, week_monday,
                lead_time_days=EMERGENCY_LEAD_TIME_DAYS,
            )
            if emergency_actions else pd.DataFrame()
        )
        reg_sched = (
            consolidate_and_schedule(regular_actions, week_monday)
            if regular_actions else pd.DataFrame()
        )
        parts = [s for s in [em_sched, reg_sched] if not s.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _week_actuals(self, week_monday: pd.Timestamp,
                      week_end: pd.Timestamp) -> pd.DataFrame:
        return self._car_events[
            self._car_events['event_date'].between(week_monday, week_end) &
            self._car_events['event_type'].isin(_ORGANIC_TYPES)
        ].copy()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, initial_state: SimulationState,
            start_date, end_date) -> None:
        """Run the simulation and populate self.logger with results."""
        self.logger.reset()
        handover_queue: dict = {}
        state   = initial_state
        cap_map = self._compounds.set_index('compound_id')['total_capacity'].to_dict()

        for week_monday in pd.date_range(start_date, end_date, freq='W-MON'):
            week_end = week_monday + pd.Timedelta(days=6)

            # 1. Refresh monthly forecast when month rolls over
            self._forecast.refresh_if_needed(
                week_monday, self._car_events, self._demand, self._compounds
            )
            multi_fc = self._build_multi_fc(week_monday)

            # 2. Emergency dispatch: compounds already at >= 99%
            emergency_actions    = self._emergency.dispatch(state.stock)
            stock_post_emergency = _apply_moves(state.stock, emergency_actions)

            # 3. Repair state for this week
            repair_load, repair_queue = self._repair_tracker.get_state(
                week_monday, state.stock
            )

            # 4. Horizon overflow plan on post-emergency stock
            projected        = HorizonProjector().project(
                stock_post_emergency, multi_fc, self._compounds,
                self.horizon_weeks, self._forecast.remarketing_rates,
            )
            overflow_actions = RouteBatcher().batch(
                OverflowCalendar().build(projected), projected,
                self._compounds, self._dist_matrix, self.urgency_window,
            )

            # 5. Repair batching (cross-compound, full trucks only)
            repair_actions = self._get_repair_actions(
                repair_load, repair_queue, projected, multi_fc, week_monday, state.stock
            )

            # 6. Pack all moves into a truck schedule respecting caps
            schedule = self._build_schedule(
                emergency_actions, overflow_actions + repair_actions, week_monday
            )

            # 6b. Append actual historical non-overflow transfers directly —
            #     bypasses consolidate_and_schedule so partial loads are preserved
            #     as-is (not repacked or dropped by floor division).
            actual_actions = self._actual_transfer_actions(week_monday, week_end)
            if actual_actions:
                actual_df = pd.DataFrame(actual_actions).assign(dispatch_date=week_monday)
                schedule = (
                    pd.concat([schedule, actual_df], ignore_index=True)
                    if not schedule.empty else actual_df
                )

            # 7. Actual organic events this week
            week_actuals = self._week_actuals(week_monday, week_end)

            # 8. Handover shortfall tracking
            ho_demand = (
                week_actuals[week_actuals['event_type'] == 'departure_handover']
                .groupby('compound_id').size().to_dict()
            )
            handover_queue = self.logger.log_handover(
                week_monday, ho_demand, state,
                repair_load, repair_queue, handover_queue,
            )

            # 9. Advance stock
            stock_new = update_stock(state.stock, week_actuals, schedule)

            # 10. Log week summary
            self.logger.log_week(
                week_monday, stock_new, schedule, cap_map,
                sum(handover_queue.values()),
            )

            state = SimulationState(
                stock        = stock_new,
                repair_queue = repair_queue,
                repair_load  = repair_load,
                week         = week_monday + pd.Timedelta(weeks=1),
            )

        self._final_stock = state.stock

    # ── Result accessors (delegate to logger) ─────────────────────────────────

    def summary(self)     -> str:            return self.logger.summary()
    def week_df(self)     -> pd.DataFrame:   return self.logger.week_df()
    def cost_df(self)     -> pd.DataFrame:   return self.logger.cost_df()
    def handover_df(self) -> pd.DataFrame:   return self.logger.handover_df()
    def final_stock(self) -> dict:           return self._final_stock
