"""
overflow.py
-----------
Overflow dispatch pipeline: two classes used in sequence.

  OverflowCalendar — identifies the FIRST week each compound will overflow
                     and the surplus at that point.

  RouteBatcher     — converts the overflow calendar into full-truckload
                     dispatch actions, filtered to an urgency window.

  _batch_dispatch  — shared greedy fill loop used by both RouteBatcher and
                     RepairBatcher (imported from repair.py).
"""

import numpy as np
import pandas as pd

from src.config import (
    OVERFLOW_THRESHOLD,
    TRUCK_CAPACITY_CARS,
    TRUCK_BASE_FEE_EUR,
    TRUCK_PER_KM_EUR,
)


def _batch_dispatch(
    sorted_senders: list,
    n_cars_map: dict,
    remaining_spare: dict,
    region_map: dict,
    dist_matrix: dict,
    reason: str,
) -> list:
    """
    Greedy full-truckload dispatch loop shared by RouteBatcher and RepairBatcher.

    Parameters
    ----------
    sorted_senders  : [(compound_id, payload), ...] — already sorted by caller.
                      payload is passed through to n_cars_map lookup.
    n_cars_map      : {compound_id: n_cars_to_dispatch}
    remaining_spare : {compound_id: spare_capacity} — mutated in place as trucks
                      are allocated; shared across all senders so no receiver is
                      double-filled.
    region_map      : {compound_id: region}
    dist_matrix     : {(from_id, to_id): km}
    reason          : value written into each action dict's 'reason' field.

    Returns
    -------
    List of action dicts: from_compound, to_compound, cars, trucks,
    distance_km, cost_eur, reason.
    """
    actions = []
    for cpd, _ in sorted_senders:
        remaining = n_cars_map[cpd]
        region    = region_map.get(cpd)

        for same_region in (True, False):
            if remaining < TRUCK_CAPACITY_CARS:
                break
            candidates = sorted(
                [
                    (c, spare, dist_matrix.get((cpd, c), np.inf))
                    for c, spare in remaining_spare.items()
                    if spare >= TRUCK_CAPACITY_CARS
                    and c != cpd
                    and (region_map.get(c) == region) == same_region
                ],
                key=lambda x: x[2],
            )
            for receiver, spare, dist_km in candidates:
                if remaining < TRUCK_CAPACITY_CARS:
                    break
                send        = min(remaining, int(spare))
                n_trucks    = send // TRUCK_CAPACITY_CARS
                if n_trucks == 0:
                    continue
                actual_send = n_trucks * TRUCK_CAPACITY_CARS
                cost        = round(
                    (TRUCK_BASE_FEE_EUR + TRUCK_PER_KM_EUR * dist_km) * n_trucks, 2
                )
                actions.append(dict(
                    from_compound = cpd,
                    to_compound   = receiver,
                    cars          = actual_send,
                    trucks        = n_trucks,
                    distance_km   = round(dist_km, 1),
                    cost_eur      = cost,
                    reason        = reason,
                ))
                remaining_spare[receiver] -= actual_send
                remaining                 -= actual_send

    return actions


class OverflowCalendar:
    """
    For each compound, find the earliest horizon week where projected stock
    exceeds the overflow threshold and compute the surplus at that point.

    Returning the FIRST overflow week (not peak) means the batcher only
    dispatches urgently — cars with later deadlines are deferred to the
    next horizon run, accumulating into full truckloads.
    """

    def build(self, projected: pd.DataFrame,
              threshold: float = OVERFLOW_THRESHOLD) -> dict:
        """
        Parameters
        ----------
        projected : output of HorizonProjector.project()

        Returns
        -------
        {compound_id: {'first_overflow_week': int, 'surplus': int}}
        Only compounds that will overflow within the horizon are included.
        """
        result = {}
        for cpd, grp in projected.groupby('compound_id'):
            grp = grp.sort_values('week_offset')
            overflow_rows = grp[grp['projected_stock'] > threshold * grp['capacity']]
            if overflow_rows.empty:
                continue
            first   = overflow_rows.iloc[0]
            surplus = int(first['projected_stock'] - threshold * first['capacity'])
            result[cpd] = {
                'first_overflow_week': int(first['week_offset']),
                'surplus':             max(0, surplus),
            }
        return result


class RouteBatcher:
    """
    For each urgent overflow compound (deadline <= urgency_window):
      1. Find the nearest receiver with projected spare capacity.
      2. Send as many full truckloads as possible (floor division).
      3. Track receiver spare across all senders.
    """

    def batch(self,
              overflow_calendar: dict,
              projected: pd.DataFrame,
              compounds_df: pd.DataFrame,
              dist_matrix: dict,
              urgency_window: int = 2) -> list:
        """
        Parameters
        ----------
        overflow_calendar : output of OverflowCalendar.build()
        projected         : output of HorizonProjector.project()
        urgency_window    : only dispatch if first_overflow_week <= this

        Returns
        -------
        List of action dicts compatible with consolidate_and_schedule.
        """
        urgent = {
            cpd: info
            for cpd, info in overflow_calendar.items()
            if info['first_overflow_week'] <= urgency_window
        }
        if not urgent:
            return []

        region_map = compounds_df.set_index('compound_id')['region'].to_dict()
        cap_map    = compounds_df.set_index('compound_id')['total_capacity'].to_dict()

        peak_stock = projected.groupby('compound_id')['projected_stock'].max().to_dict()
        remaining_spare = {
            cpd: max(0, int(OVERFLOW_THRESHOLD * cap_map.get(cpd, 0) - peak_stock.get(cpd, 0)))
            for cpd in cap_map
            if cpd not in urgent
            and cap_map.get(cpd, 0) > 0
            and peak_stock.get(cpd, 0) < OVERFLOW_THRESHOLD * cap_map[cpd]
        }

        week0_stock = (
            projected[projected['week_offset'] == 0]
            .set_index('compound_id')['projected_stock'].to_dict()
        )
        week0_util = {
            cpd: week0_stock.get(cpd, 0) / cap_map[cpd]
            for cpd in cap_map if cap_map.get(cpd, 0) > 0
        }

        sorted_senders = sorted(
            urgent.items(),
            key=lambda x: (
                x[1]['first_overflow_week'],
                -week0_util.get(x[0], 0),
                -x[1]['surplus'],
            ),
        )

        n_cars_map = {cpd: info['surplus'] for cpd, info in urgent.items()}
        return _batch_dispatch(
            sorted_senders, n_cars_map, remaining_spare,
            region_map, dist_matrix, reason='horizon_overflow',
        )
