"""FWI processing package for gridded Canadian Fire Weather Index workflows."""

from .config import AppConfig, load_config
from .fwi_algorithm import ComputationResult, FWIState, compute_fwi_indices

__all__ = ["AppConfig", "ComputationResult", "FWIState", "compute_fwi_indices", "load_config"]

__version__ = "0.1.0"