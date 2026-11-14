"""SynPV - physics-based synthetic residential PV fleet generator.

Hourly NASA POWER meteorology -> advecting cloud field -> heterogeneous fleet
of rooftop PV systems (CEC modules and inverters) -> pvlib/SAM model chain ->
5-min ideal and measured AC power with ground-truth variables.
"""
__version__ = "1.0.0"

from .config import DEFAULT_CONFIG, default_config, load_config, paper_config  # noqa: E402
from .simulator import Simulation, SimulationCancelled, SimulationResult, run  # noqa: E402
from .weather import WeatherError, fetch_nasa_power  # noqa: E402

__all__ = ["__version__", "DEFAULT_CONFIG", "default_config", "paper_config", "load_config",
           "Simulation", "SimulationResult", "SimulationCancelled", "run",
           "WeatherError", "fetch_nasa_power"]
