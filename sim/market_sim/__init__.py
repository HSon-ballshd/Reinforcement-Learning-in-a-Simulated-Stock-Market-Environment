"""Cookie Clicker stock-market simulator (Python port).

Implements the interface defined in collaboration/CONTRACT.md (v1):
- sim.market_sim.simulator.CookieClickerMarket
- sim.market_sim.dataset.generate_regime_dataset
"""
from .simulator import CookieClickerMarket
from .dataset import generate_regime_dataset

__all__ = ["CookieClickerMarket", "generate_regime_dataset"]