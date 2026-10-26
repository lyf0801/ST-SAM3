from .OnlySup import OnlySup

from .CPS import CPS
from .CCT import CCT
from .CutMix import CutMix
from .DebiasPL import DebiasPL
from .FixMatch import FixMatch, NFixMatch
from .UniMatch import UniMatch
from .AdaptMatch import AdaptMatch

from .DBMatch import DBMatch, NDBMatch


def create_algorithm(args, device):
    """Return the algorithm class with the given name."""
    if args.algorithm not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(args.algorithm))
    return globals()[args.algorithm](args, device)
