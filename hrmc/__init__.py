"""RMC and hybrid RMC (HRMC) refinement of amorphous structures against neutron G(r)."""
from . import rmc, hrmc
from .io import read_gr

__all__ = ["rmc", "hrmc", "read_gr"]
