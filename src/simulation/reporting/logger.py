"""
logger.py
---------
SimulationLogger — accumulates per-week metrics and exposes results
as DataFrames after the simulation run completes.

The runner calls:
  logger.reset()
  logger.log_handover(week, ho_demand, state, repair_load, repair_queue, queue) → new_queue
  logger.log_week(week, stock_new, schedule, cap_map, queue_total)

The notebook calls:
  runner.logger.week_df()
  runner.logger.cost_df()
  runner.logger.handover_df()
  runner.logger.summary()
"""

import pandas as pd


class SimulationLogger:

    def __init__(self):
        self._log_weeks:    list = []
        self._log_schedule: list = []
        self._log_handover: list = []

    def reset(self) -> None:
        self._log_weeks    = []
        self._log_schedule = []
        self._log_handover = []

    # ── Per-week logging ─────────────────────────────────────────────────────

    def log_week(self,
                 week:         pd.Timestamp,
                 stock_new:    dict,
                 schedule:     pd.DataFrame,
                 cap_map:      dict,
                 queue_total:  int) -> None:
        overflow_count = sum(
            1 for cpd, s in stock_new.items()
            if cap_map.get(cpd, 0) > 0 and s / cap_map[cpd] >= 0.9
        )
        self._log_weeks.append(dict(
            week                 = week,
            trucks               = int(schedule['trucks'].sum()) if not schedule.empty else 0,
            cost_eur             = float(schedule['cost_eur'].sum()) if not schedule.empty else 0.0,
            total_stock          = sum(stock_new.values()),
            overflow_compounds   = overflow_count,
            handover_queue_total = queue_total,
        ))
        if not schedule.empty:
            for _, row in schedule.iterrows():
                self._log_schedule.append(row.to_dict() | {'week': week})

    def log_handover(self,
                     week:          pd.Timestamp,
                     ho_demand:     dict,
                     state,
                     repair_load:   dict,
                     repair_queue:  dict,
                     handover_queue: dict) -> dict:
        """
        Compute per-compound handover shortfall, log it, and return the
        updated carry-forward queue for the next week.
        """
        new_queue = {}
        all_cpds  = set(ho_demand) | set(handover_queue)

        for cpd in all_cpds:
            this_week = ho_demand.get(cpd, 0)
            backlog   = handover_queue.get(cpd, 0)
            effective = this_week + backlog
            avail     = max(0, (
                state.stock.get(cpd, 0)
                - repair_load.get(cpd, 0)
                - repair_queue.get(cpd, 0)
            ))
            served    = min(avail, effective)
            shortfall = max(0, effective - served)
            new_queue[cpd] = shortfall

            self._log_handover.append(dict(
                week             = week,
                compound_id      = cpd,
                available        = avail,
                demand           = this_week,
                queue_backlog    = backlog,
                effective_demand = effective,
                shortfall        = shortfall,
            ))

        return new_queue

    # ── Results ──────────────────────────────────────────────────────────────

    def week_df(self) -> pd.DataFrame:
        """Weekly summary: week, trucks, cost_eur, total_stock, overflow_compounds."""
        return pd.DataFrame(self._log_weeks)

    def cost_df(self) -> pd.DataFrame:
        """All scheduled dispatches with a week column."""
        return pd.DataFrame(self._log_schedule) if self._log_schedule else pd.DataFrame()

    def handover_df(self) -> pd.DataFrame:
        """Per-compound per-week handover check: available, demand, shortfall."""
        return pd.DataFrame(self._log_handover) if self._log_handover else pd.DataFrame(
            columns=['week', 'compound_id', 'available', 'demand', 'shortfall']
        )

    def summary(self) -> str:
        df  = self.cost_df()
        hdf = self.handover_df()
        shortfall_weeks = int((hdf.groupby('week')['shortfall'].sum() > 0).sum()) if not hdf.empty else 0
        total_shortfall = int(hdf['shortfall'].sum()) if not hdf.empty else 0
        lines = [
            f"Total trucks : {int(df['trucks'].sum()) if not df.empty else 0}",
            f"Total cost   : €{df['cost_eur'].sum():,.2f}" if not df.empty else "Total cost   : €0.00",
            f"Overflow-risk compound-weeks : {int(self.week_df()['overflow_compounds'].sum())}",
            f"Handover shortfall weeks     : {shortfall_weeks}",
            f"Total cars short for handover: {total_shortfall}",
        ]
        return "\n".join(lines)
