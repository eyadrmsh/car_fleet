"""
src/simulation
--------------
Week-by-week compound network simulation.

Quick start:
    from src.simulation import SimulationState, HorizonSimulationRunner
    from src.simulation.stock import (
        apply_repair_fifo, calibrate_initial_stock,
        build_stock_series, assess_compound_status,
    )
"""

from .state import SimulationState
from .runner import HorizonSimulationRunner
from .stock import (
    apply_repair_fifo,
    calibrate_initial_stock,
    build_stock_series,
    assess_compound_status,
)
from .dispatch.scheduler import consolidate_and_schedule
