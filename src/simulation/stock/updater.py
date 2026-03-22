"""
updater.py
----------
update_stock — apply one week of organic net flow + scheduled transfers.
"""

import pandas as pd


_SIGN_MAP = {
    'arrival_return':        1,
    'arrival_new_stock':     1,
    'departure_handover':   -1,
    'departure_remarketing': -1,
}


def update_stock(prev_stock: dict,
                 week_actuals: pd.DataFrame,
                 schedule: pd.DataFrame) -> dict:
    """
    Apply organic net flow (arrivals − departures) and scheduled truck moves.
    Transfer events are handled via schedule; actuals covers organic flows only.
    Stock is floored at 0.
    """
    organic_net = (
        week_actuals
        .assign(sign=week_actuals['event_type'].map(_SIGN_MAP))
        .groupby('compound_id')['sign']
        .sum()
        .to_dict()
    )

    stock = {
        cpd: max(0, int(val) + organic_net.get(cpd, 0))
        for cpd, val in prev_stock.items()
    }

    if not schedule.empty:
        for _, row in schedule.iterrows():
            stock[row['from_compound']] = max(0, stock[row['from_compound']] - row['cars'])
            stock[row['to_compound']]   = stock.get(row['to_compound'], 0) + row['cars']

    return stock
