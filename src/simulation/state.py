"""
state.py
--------
SimulationState — snapshot of the compound network at the start of a week.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SimulationState:
    """
    Snapshot of the compound network entering a given week.

    Attributes
    ----------
    stock           : {compound_id: int} — total cars at each compound
    repair_queue    : {compound_id: int} — non-repair compounds only:
                      cars waiting to be dispatched to a repair compound
    repair_load     : {compound_id: int} — repair compounds only:
                      cars currently occupying repair bays
    handover_queue  : {compound_id: int} — customers waiting for a car
                      because available stock was insufficient last week;
                      accumulated across weeks until stock recovers
    week            : Monday of the week this state represents
    """
    stock:           dict
    repair_queue:    dict
    repair_load:     dict
    week:            pd.Timestamp
    handover_queue:  dict = field(default_factory=dict)

    def available_stock(self) -> dict:
        """
        Cars not locked in repair bays or waiting for repair dispatch.
        Used by run_weekly_rules to determine rebalancing donor/shortfall.
        """
        return {
            cpd: max(
                0,
                self.stock.get(cpd, 0)
                - self.repair_load.get(cpd, 0)
                - self.repair_queue.get(cpd, 0),
            )
            for cpd in self.stock
        }
