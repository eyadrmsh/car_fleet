"""
repair_tracker.py
-----------------
RepairStateTracker — precomputes FIFO bay occupancy from pre-simulation
car events and returns (repair_load, repair_queue) for any given week.

repair_load  : {compound_id: n} — cars actively in bays at repair-capable compounds.
               Derived from FIFO-assigned repair_start/end dates.
repair_queue : {compound_id: n} — cars at non-repair compounds estimated to need
               repair dispatch. Derived from stock × historical repair share.
"""

import pandas as pd

from src.preprocessing import compute_repair_share


class RepairStateTracker:
    """
    Initialised once from pre-simulation FIFO-scheduled events.
    Call get_state(week_monday, stock) each week in the simulation loop.
    """

    def __init__(self, ev_train_sched: pd.DataFrame, compounds: pd.DataFrame):
        repair_capable = set(
            compounds.loc[
                compounds['repair_capable'].astype(str).str.lower().isin(['true', '1', 'yes']),
                'compound_id',
            ]
        )
        self._bay_cap = (
            compounds.set_index('compound_id')['repair_capacity_per_week'].to_dict()
        )
        self._non_repair = set(compounds['compound_id']) - repair_capable

        in_bays = ev_train_sched[
            ev_train_sched['repair_start_date'].notna() &
            ev_train_sched['repair_end_date'].notna() &
            ev_train_sched['compound_id'].isin(repair_capable)
        ][['compound_id', 'repair_start_date', 'repair_end_date']].copy()
        in_bays['repair_start_date'] = pd.to_datetime(in_bays['repair_start_date'])
        in_bays['repair_end_date']   = pd.to_datetime(in_bays['repair_end_date'])
        self._bay_cars = in_bays

        shares = compute_repair_share(ev_train_sched)
        self._repair_share        = shares.set_index('compound_id')['repair_share'].to_dict()
        self._global_repair_share = float(shares['repair_share'].mean())

    def get_state(self, week_monday: pd.Timestamp, stock: dict) -> tuple:
        """
        Return (repair_load, repair_queue) for the given week.

        repair_load  — cars in bays this week, capped by simultaneous bay capacity.
        repair_queue — estimated repair backlog at non-repair compounds.
        """
        active = self._bay_cars[
            (self._bay_cars['repair_start_date'] <= week_monday) &
            (self._bay_cars['repair_end_date']   >= week_monday)
        ]
        repair_load = {
            cpd: min(len(grp), self._bay_cap.get(cpd, len(grp)))
            for cpd, grp in active.groupby('compound_id')
        }
        repair_queue = {
            cpd: int(round(
                stock.get(cpd, 0)
                * self._repair_share.get(cpd, self._global_repair_share)
            ))
            for cpd in self._non_repair
        }
        return repair_load, repair_queue
