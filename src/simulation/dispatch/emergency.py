"""
emergency.py
------------
EmergencyDispatcher — immediately moves cars from any compound at >= 99%
capacity. Bypasses the horizon plan; fires with no lead time.
"""

import math

import numpy as np
import pandas as pd

from src.config import (
    EMERGENCY_THRESHOLD,
    EMERGENCY_SURCHARGE,
    OVERFLOW_THRESHOLD,
    TRUCK_CAPACITY_CARS,
    TRUCK_BASE_FEE_EUR,
    TRUCK_PER_KM_EUR,
)


class EmergencyDispatcher:
    """
    Dispatch cars immediately from any compound at >= EMERGENCY_THRESHOLD (99%).
    Uses ceil trucks (cannot defer) and EMERGENCY_SURCHARGE (2×) cost.
    Receivers must be below OVERFLOW_THRESHOLD in current stock.
    Senders are processed most-full first; same-region receivers preferred.
    """

    def __init__(self, compounds: pd.DataFrame, dist_matrix: dict):
        self._compounds   = compounds
        self._dist_matrix = dist_matrix

    def dispatch(self, stock: dict) -> list:
        """Return list of action dicts, empty if no compound is at emergency level."""
        cap_map    = self._compounds.set_index('compound_id')['total_capacity'].to_dict()
        region_map = self._compounds.set_index('compound_id')['region'].to_dict()

        senders = {
            cpd: stk
            for cpd, stk in stock.items()
            if cap_map.get(cpd, 0) > 0 and stk / cap_map[cpd] >= EMERGENCY_THRESHOLD
        }
        if not senders:
            return []

        receiver_spare = {
            cpd: max(0, int(OVERFLOW_THRESHOLD * cap_map[cpd] - stock.get(cpd, 0)))
            for cpd in cap_map
            if cpd not in senders
            and cap_map.get(cpd, 0) > 0
            and stock.get(cpd, 0) / cap_map[cpd] < OVERFLOW_THRESHOLD
        }

        actions = []
        for cpd, stk in sorted(senders.items(), key=lambda x: -x[1] / cap_map[x[0]]):
            surplus = int(stk - OVERFLOW_THRESHOLD * cap_map[cpd])
            region  = region_map.get(cpd)

            for same_region in (True, False):
                if surplus <= 0:
                    break
                candidates = sorted(
                    [
                        (c, spare, self._dist_matrix.get((cpd, c), np.inf))
                        for c, spare in receiver_spare.items()
                        if spare >= TRUCK_CAPACITY_CARS and c != cpd
                        and (region_map.get(c) == region) == same_region
                    ],
                    key=lambda x: x[2],
                )
                for receiver, spare, dist_km in candidates:
                    if surplus <= 0:
                        break
                    send     = min(surplus, int(spare))
                    n_trucks = math.ceil(send / TRUCK_CAPACITY_CARS)
                    if n_trucks == 0:
                        continue
                    actual_send = n_trucks * TRUCK_CAPACITY_CARS
                    cost = round(
                        (TRUCK_BASE_FEE_EUR + TRUCK_PER_KM_EUR * dist_km)
                        * n_trucks * EMERGENCY_SURCHARGE, 2
                    )
                    actions.append(dict(
                        from_compound = cpd,
                        to_compound   = receiver,
                        cars          = actual_send,
                        trucks        = n_trucks,
                        distance_km   = round(dist_km, 1),
                        cost_eur      = cost,
                        reason        = 'emergency_overflow',
                    ))
                    receiver_spare[receiver] -= actual_send
                    surplus                  -= actual_send

        return actions
