"""
status.py
---------
assess_compound_status — snapshot of every compound's repair state
(available, in_repair, waiting_repair) at a given date.
"""

import pandas as pd

from src.preprocessing import compute_repair_share
from .repair_fifo import apply_repair_fifo


def assess_compound_status(events: pd.DataFrame,
                            compounds: pd.DataFrame,
                            as_of_date,
                            stock: dict = None,
                            repair_queue: dict = None) -> pd.DataFrame:
    """
    Return a DataFrame with status for every compound on as_of_date.

    Columns: compound_id, total_capacity, repair_capable, total,
             available, in_repair, waiting_repair, utilisation.

    repair_queue : optional {compound_id: int} explicit waiting_repair counts
                  for non-repair compounds. When provided, used directly instead
                  of share-based estimation.
    """
    as_of = pd.Timestamp(as_of_date)
    arrival_types = {'arrival_return', 'arrival_new_stock', 'arrival_transfer_in'}

    all_ev = events[events['event_date'] <= as_of].copy()
    repair_shares = compute_repair_share(all_ev)
    share_map    = repair_shares.set_index('compound_id')['repair_share'].to_dict()
    global_share = float(repair_shares['repair_share'].mean())

    all_ev['event_date'] = pd.to_datetime(all_ev['event_date'])
    all_ev = apply_repair_fifo(all_ev, compounds)

    last_ev = (
        all_ev.sort_values(['car_id', 'event_date'])
              .groupby('car_id').last()
              .reset_index()
    )

    rows = []
    for _, cpd_row in compounds.iterrows():
        cpd               = cpd_row['compound_id']
        cap               = int(cpd_row['total_capacity'])
        repair_cap        = int(cpd_row['repair_capacity_per_week'])
        is_repair_capable = str(cpd_row['repair_capable']).lower() in ['true', '1', 'yes']
        total             = float(stock.get(cpd, 0)) if stock else 0.0

        present = last_ev[
            (last_ev['compound_id'] == cpd) &
            (last_ev['event_type'].isin(arrival_types))
        ]

        n_in_repair = 0
        n_waiting   = 0

        if is_repair_capable:
            for _, row in present.iterrows():
                if not bool(row.get('needs_repair', False)):
                    continue
                start = row.get('repair_start_date')
                end   = row.get('repair_end_date')
                if pd.isna(end):
                    n_waiting += 1
                elif pd.Timestamp(start) > as_of:
                    n_waiting += 1
                elif pd.Timestamp(start) < as_of < pd.Timestamp(end):
                    n_in_repair += 1
            if repair_cap > 0:
                n_in_repair = min(n_in_repair, repair_cap)
        else:
            if repair_queue is not None:
                n_waiting = repair_queue.get(cpd, 0)
            else:
                n_waiting = int(round(total * share_map.get(cpd, global_share)))

        available = max(0, int(total) - n_in_repair - n_waiting)

        rows.append(dict(
            compound_id    = cpd,
            total_capacity = cap,
            repair_capable = is_repair_capable,
            total          = int(total),
            available      = available,
            in_repair      = n_in_repair,
            waiting_repair = n_waiting,
            utilisation    = round(total / cap, 3) if cap > 0 else 0.0,
        ))

    return pd.DataFrame(rows)
