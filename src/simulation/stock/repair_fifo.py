"""
repair_fifo.py
--------------
FIFO bay scheduling: assigns repair_start_date / repair_end_date to every
car that needs repair at a repair-capable compound.
"""

import heapq

import pandas as pd


def apply_repair_fifo(car_events: pd.DataFrame,
                      compounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Schedule repair for all cars with needs_repair=True at repair-capable
    compounds. Uses FIFO order and respects repair_capacity_per_week
    (simultaneous bay slots, not weekly throughput).

    Adds repair_start_date and repair_end_date columns to the returned DataFrame.
    """
    repair_caps = (
        compounds_df[
            compounds_df['repair_capable'].astype(str).str.lower().isin(['true', '1', 'yes'])
        ]
        .set_index('compound_id')['repair_capacity_per_week']
        .to_dict()
    )

    ev = car_events.copy()
    ev['event_date'] = pd.to_datetime(ev['event_date'])
    ev['repair_start_date'] = pd.NaT
    ev['repair_end_date']   = pd.NaT

    arrival_types = {'arrival_return', 'arrival_transfer_in'}

    for compound_id, capacity in repair_caps.items():
        if capacity <= 0:
            continue

        mask = (
            (ev['compound_id'] == compound_id) &
            (ev['event_type'].isin(arrival_types)) &
            (ev['needs_repair'] == True)
        )
        repair_cars = ev[mask].sort_values('event_date')

        active = []
        for idx, row in repair_cars.iterrows():
            arrival     = row['event_date']
            repair_days = float(row['estimated_repair_days'])

            while active and active[0] <= arrival.value:
                heapq.heappop(active)

            if len(active) < capacity:
                repair_start = arrival
            else:
                repair_start = pd.Timestamp(heapq.heappop(active))

            repair_end = repair_start + pd.Timedelta(days=repair_days)
            heapq.heappush(active, repair_end.value)

            ev.at[idx, 'repair_start_date'] = repair_start
            ev.at[idx, 'repair_end_date']   = repair_end

    return ev
