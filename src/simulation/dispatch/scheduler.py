"""
scheduling.py
-------------
Group weekly dispatch actions into a truck schedule, respecting daily network caps.

Entry point:
    schedule = consolidate_and_schedule(week_actions, week_monday)
    # Returns DataFrame: dispatch_date, from_compound, to_compound,
    #                    trucks, cars, distance_km, cost_eur, reason
"""

import pandas as pd

from src.config import (
    TRUCK_CAPACITY_CARS, TRUCK_BASE_FEE_EUR, TRUCK_PER_KM_EUR,
    MAX_TRUCKS_PER_DAY, MAX_TRUCKS_PER_ROUTE, STANDARD_LEAD_TIME_DAYS,
)

_PRIORITY = {
    'emergency_overflow':  -2,
    'emergency_handover':  -1,
    'overflow_dispatch':    0,
    'demand_rebalancing':   1,
    'repair_routing':       2,
}
_WORKDAYS  = 5


def consolidate_and_schedule(week_actions: list,
                              week_monday: pd.Timestamp,
                              lead_time_days: int = STANDARD_LEAD_TIME_DAYS) -> pd.DataFrame:
    """
    Group route actions by (from_compound, to_compound), then spread trucks
    across available workdays respecting:
      - MAX_TRUCKS_PER_ROUTE per route per day
      - MAX_TRUCKS_PER_DAY network-wide per day
      - STANDARD_LEAD_TIME_DAYS minimum booking lead time

    Higher-priority reasons (overflow_dispatch > repair_routing > demand_rebalancing)
    are scheduled first and therefore get earlier dispatch slots.

    Parameters
    ----------
    week_actions : list of action dicts from run_weekly_rules()
    week_monday  : Monday of the dispatch week

    Returns
    -------
    DataFrame with columns:
        dispatch_date, from_compound, to_compound, trucks, cars,
        distance_km, cost_eur, reason
    """
    if not week_actions:
        return pd.DataFrame()

    wa = pd.DataFrame(week_actions)
    wa['priority'] = wa['reason'].map(_PRIORITY).fillna(9)
    wa = wa.sort_values('priority')

    # Aggregate to one row per route (same pair may appear from different rules)
    route_grp = (
        wa.groupby(['from_compound', 'to_compound'])
        .agg(
            cars        = ('cars',        'sum'),
            distance_km = ('distance_km', 'first'),
            reason      = ('reason',      'first'),   # highest-priority reason
            priority    = ('priority',    'min'),
        )
        .reset_index()
        .sort_values('priority')
    )
    route_grp['trucks'] = route_grp['cars'] // TRUCK_CAPACITY_CARS
    route_grp = route_grp[route_grp['trucks'] > 0]

    first_day = week_monday + pd.Timedelta(days=lead_time_days)
    workdays  = [
        week_monday + pd.Timedelta(days=d)
        for d in range(_WORKDAYS)
        if week_monday + pd.Timedelta(days=d) >= first_day
    ]

    day_trucks       = {d: 0 for d in workdays}
    day_route_trucks = {}
    schedule         = []

    unscheduled = []   # trucks that could not fit within the weekly cap

    for _, row in route_grp.iterrows():
        trucks_left = int(row['trucks'])
        for day in workdays:
            if trucks_left <= 0:
                break
            route_key = (day, row['from_compound'], row['to_compound'])
            already   = day_route_trucks.get(route_key, 0)
            can_send  = min(
                trucks_left,
                MAX_TRUCKS_PER_ROUTE - already,
                MAX_TRUCKS_PER_DAY   - day_trucks[day],
            )
            if can_send <= 0:
                continue
            day_trucks[day]             += can_send
            day_route_trucks[route_key]  = already + can_send
            trucks_left                 -= can_send
            schedule.append(dict(
                dispatch_date = day,
                from_compound = row['from_compound'],
                to_compound   = row['to_compound'],
                trucks        = can_send,
                cars          = can_send * TRUCK_CAPACITY_CARS,
                distance_km   = row['distance_km'],
                cost_eur      = round(
                    (TRUCK_BASE_FEE_EUR + TRUCK_PER_KM_EUR * row['distance_km']) * can_send, 2
                ),
                reason        = row['reason'],
            ))

        if trucks_left > 0:
            unscheduled.append(
                f"  {row['from_compound']} → {row['to_compound']}"
                f" ({row['reason']}): {trucks_left} truck(s) / "
                f"{trucks_left * TRUCK_CAPACITY_CARS} cars could not be scheduled"
            )

    if unscheduled:
        print(
            f"WARNING [{week_monday.date()}]: {len(unscheduled)} route(s) hit the "
            f"{MAX_TRUCKS_PER_DAY}-truck/day cap — moves deferred to next week:\n"
            + "\n".join(unscheduled)
        )

    return pd.DataFrame(schedule) if schedule else pd.DataFrame()
