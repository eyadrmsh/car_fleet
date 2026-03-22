"""
repair.py
---------
RepairBatcher — consolidates pending repair dispatches across all non-repair
compounds sending to the same repair destination.

Uses peak projected stock (from HorizonProjector) to assess receiver eligibility
across the full horizon — a repair compound that is fine today but will overflow
in week 2 is excluded.

The dispatch loop itself is shared with RouteBatcher via _batch_dispatch
(overflow.py).  This class only differs in how senders and the receiver pool
are constructed:
  - Senders   : non-repair compounds with repair_queue >= TRUCK_CAPACITY_CARS
  - Receivers : repair-capable compounds only (vs any non-overflowing compound)
  - Extra cap : sender keeps enough cars to cover this week's handover forecast
"""

import pandas as pd

from src.config import (
    OVERFLOW_THRESHOLD,
    TRUCK_CAPACITY_CARS,
)
from src.simulation.dispatch.overflow import _batch_dispatch


class RepairBatcher:
    """
    For each non-repair compound with >= TRUCK_CAPACITY_CARS repair cars queued:
      1. Find the nearest eligible repair-capable compound (same region first).
      2. Eligible = peak projected stock across the horizon stays below
         OVERFLOW_THRESHOLD — same check as RouteBatcher uses for overflow.
      3. Dispatch as many full truckloads as possible (floor division).
      4. Track remaining_spare per receiver across all senders to prevent
         double-filling a single repair compound.
    """

    def batch(
        self,
        repair_queue: dict,
        projected: pd.DataFrame,
        repair_load: dict,
        compounds_df: pd.DataFrame,
        dist_matrix: dict,
        available_stock: dict = None,
        ho_weekly_fc: dict = None,
    ) -> list:
        """
        Parameters
        ----------
        repair_queue    : {compound_id: n_cars} — repair cars queued at non-repair
                          compounds.
        projected       : HorizonProjector.project() output — reused from this
                          week's overflow projection for peak stock check.
        repair_load     : {compound_id: n} — cars currently occupying repair bays
                          (informational; routing uses stock capacity not bay count).
        compounds_df    : compound metadata (compound_id, total_capacity,
                          repair_capable, region)
        dist_matrix     : {(from_id, to_id): km}
        available_stock : optional {compound_id: n} — stock − repair_load −
                          repair_queue.  When provided, caps dispatch so the sender
                          retains at least ho_weekly_fc cars for handovers.
        ho_weekly_fc    : optional {compound_id: weekly_handover_forecast}

        Returns
        -------
        List of action dicts compatible with consolidate_and_schedule.
        """
        _avail = available_stock or {}
        _ho_fc = ho_weekly_fc or {}

        # Cap each sender so it keeps at least 1× weekly handover forecast
        # available after trucks leave.  If headroom < 1 full truck, defer.
        def _headroom(cpd, queue_n):
            if not _avail:
                return queue_n
            return max(0, int(_avail.get(cpd, queue_n) - _ho_fc.get(cpd, 0)))

        senders = {
            cpd: min(n, _headroom(cpd, n))
            for cpd, n in repair_queue.items()
            if min(n, _headroom(cpd, n)) >= TRUCK_CAPACITY_CARS
        }
        if not senders:
            return []

        cap_map    = compounds_df.set_index("compound_id")["total_capacity"].to_dict()
        region_map = compounds_df.set_index("compound_id")["region"].to_dict()

        repair_capable = set(
            compounds_df.loc[
                compounds_df["repair_capable"]
                    .astype(str).str.lower().isin(["true", "1", "yes"]),
                "compound_id",
            ]
        )

        peak_stock = projected.groupby("compound_id")["projected_stock"].max().to_dict()
        remaining_spare = {
            cpd: max(0, int(OVERFLOW_THRESHOLD * cap_map.get(cpd, 0)
                            - peak_stock.get(cpd, 0)))
            for cpd in repair_capable
            if cap_map.get(cpd, 0) > 0
            and peak_stock.get(cpd, 0) < OVERFLOW_THRESHOLD * cap_map[cpd]
        }

        # Most critical senders first: largest repair backlog dispatched first.
        sorted_senders = sorted(senders.items(), key=lambda x: -x[1])

        return _batch_dispatch(
            sorted_senders, senders, remaining_spare,
            region_map, dist_matrix, reason="repair_routing",
        )
