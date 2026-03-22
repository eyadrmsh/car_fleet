"""
projector.py
------------
HorizonProjector — projects compound stock N weeks forward using a
multi-week forecast DataFrame.

Output: DataFrame [week_offset, compound_id, projected_stock, capacity, utilisation]
"""

import pandas as pd

from src.config import OVERFLOW_THRESHOLD


class HorizonProjector:
    """
    Project compound stock levels N weeks into the future.

    At each week the net flow (arrivals - departures - remarketing) from the
    forecast is applied to the running stock.  The result shows when and by
    how much each compound will breach the overflow threshold.
    """

    def project(
        self,
        stock: dict,
        weekly_fc: pd.DataFrame,
        compounds_df: pd.DataFrame,
        horizon_weeks: int = 4,
        remarketing_rates: dict = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        stock             : {compound_id: current_stock}
        weekly_fc         : DataFrame [week_start, compound_id, event_type, weekly_forecast]
                            Must cover at least horizon_weeks distinct week_starts.
        compounds_df      : compound metadata (compound_id, total_capacity)
        horizon_weeks     : number of weeks to project ahead
        remarketing_rates : optional {compound_id: mean_weekly_cars_leaving_fleet}
                            Subtracted from stock each week as a permanent fleet drain.

        Returns
        -------
        DataFrame [week_offset, compound_id, projected_stock, capacity, utilisation]
        Week offset 0 = current stock before any flow this week.
        Week offset 1 = stock after week 1 net flow, etc.
        """
        cap_map  = compounds_df.set_index("compound_id")["total_capacity"].to_dict()
        rm_rates = remarketing_rates or {}
        weeks    = sorted(weekly_fc["week_start"].unique())[:horizon_weeks]

        arrivals_types   = {"arrival_return", "arrival_new_stock"}
        departures_types = {"departure_handover"}

        rows    = []
        running = dict(stock)

        # Week offset 0: current stock snapshot
        for cpd, s in running.items():
            cap = cap_map.get(cpd, 0)
            rows.append(dict(
                week_offset     = 0,
                compound_id     = cpd,
                projected_stock = s,
                capacity        = cap,
                utilisation     = s / cap if cap > 0 else 0.0,
            ))

        for offset, week_start in enumerate(weeks, start=1):
            wk = weekly_fc[weekly_fc["week_start"] == week_start]

            arr = (
                wk[wk["event_type"].isin(arrivals_types)]
                .groupby("compound_id")["weekly_forecast"].sum()
                .to_dict()
            )
            dep = (
                wk[wk["event_type"].isin(departures_types)]
                .groupby("compound_id")["weekly_forecast"].sum()
                .to_dict()
            )

            for cpd in running:
                net = (arr.get(cpd, 0)
                       - dep.get(cpd, 0)
                       - rm_rates.get(cpd, 0))
                running[cpd] = max(0, running[cpd] + net)
                cap = cap_map.get(cpd, 0)
                rows.append(dict(
                    week_offset     = offset,
                    compound_id     = cpd,
                    projected_stock = running[cpd],
                    capacity        = cap,
                    utilisation     = running[cpd] / cap if cap > 0 else 0.0,
                ))

        return pd.DataFrame(rows)
