#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Cellnition.**
'''

# ....................{ IMPORTS                            }....................
from beartype.claw import beartype_this_package
from warnings import filterwarnings

# ....................{ QA                                 }....................
# Enforce type hints across this package with @beartype.
beartype_this_package()

# ....................{ WARNINGS                           }....................
# Unconditionally ignore all non-fatal warnings emitted by the third-party
# scipy.optimize.fsolve() function of the forms:
#     ../conda/envs/ionyou_dev/lib/python3.11/site-packages/scipy/optimize/_minpack_py.py:177:
#     RuntimeWarning: The iteration is not making good progress, as measured by the
#     improvement from the last five Jacobian evaluations.
#       warnings.warn(msg, RuntimeWarning)
#
#     ../py/conda/envs/ionyou_dev/lib/python3.11/site-packages/scipy/optimize/_minpack_py.py:177:
#     RuntimeWarning: xtol=0.000000 is too small, no further improvement in the approximate
#     solution is possible.
#       warnings.warn(msg, RuntimeWarning)
filterwarnings(
    action='ignore',
    #FIXME: See below, please. *sigh*
    # module=r'^scipy\..*',
    category=RuntimeWarning,
    message=r'^The iteration is not making good progress',
)
filterwarnings(
    action='ignore',
    #FIXME: Ideally, this warning should be matched as originating from the
    #"scipy" package. Unfortunately, for unknown reasons, CPython claims this
    #warning originates from *OUR* package instead: e.g.,
    #    home/pietakio/py/cellnition/cellnition/science/network_models/probability_networks.py:562:
    #    RuntimeWarning: xtol=0.000000 is too small, no further improvement in the approximate
    # module=r'^scipy\..*',
    category=RuntimeWarning,
    message=r'^xtol=0\.000000 is too small,',
)

# Unconditionally ignore all non-fatal warnings emitted by the third-party
# "pygraphviz.agraph" submodule of the forms:
#     ../py/conda/envs/ionyou_dev/lib/python3.13/site-packages/pygraphviz/agraph.py:1403:
#     RuntimeWarning: Warning: some nodes with margin (3.20,3.20) touch -
#     falling back to straight line edges
#       warnings.warn(b"".join(errors).decode(self.encoding), RuntimeWarning)
filterwarnings(
    action='ignore',
    #FIXME: This *SHOULD* work, but currently doesn't. Perhaps regular
    #expressions are prohibited as the "module" parameter? No idea.
    # module=r'^pygraphviz\..*',
    category=RuntimeWarning,
    message=r'^Warning: some nodes with margin',
)

# ....................{ GLOBALS                            }....................
# Declare PEP 8-compliant version constants expected by external automation.

__version__ = '0.6.0'
'''
Human-readable application version as a ``.``-delimited string.

For :pep:`8` compliance, this specifier has the canonical name ``__version__``
rather than that of a typical global (e.g., ``VERSION_STR``).
'''