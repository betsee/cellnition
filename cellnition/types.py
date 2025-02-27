#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Project-wide **type hints** (i.e., PEP-compliant objects usable as type hints
annotating callable parameters and returns).
'''

# ....................{ IMPORTS                            }....................
from numpy import bool_

# ....................{ HINTS                              }....................
NumpyTrue = bool_(True)
'''
:mod:`numpy`-specific analogue of the :data:`True` singleton.
'''


NumpyFalse = bool_(False)
'''
:mod:`numpy`-specific analogue of the :data:`False` singleton.
'''