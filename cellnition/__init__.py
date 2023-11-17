#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2024 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Cellnition.**

For :pep:`8` compliance, this namespace exposes a subset of the metadata
constants provided by the :mod:`cellnition.meta` submodule commonly inspected
and thus expected by external automation.
'''

# ....................{ TODO                               }....................
#FIXME: [SESSION] As time permits, implement most or all of the excellent advice
#at this blog article. Although the author focuses on session auto-save and
#auto-load to improve resiliency in the face of browser timeouts and refreshes
#(which is essential functionality, really), pretty much *ALL* of the advice
#here is awesome:
#    https://towardsdatascience.com/10-features-your-streamlit-ml-app-cant-do-without-implemented-f6b4f0d66d36

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: To avoid race conditions during setuptools-based installation, this
# module may import *ONLY* from modules guaranteed to exist at the start of
# installation. This includes all standard Python modules and package-specific
# modules but *NOT* third-party dependencies, which if currently uninstalled
# will only be installed at some later time in the installation. Likewise, to
# avoid circular import dependencies, the top-level of this module should avoid
# importing package-specific modules where feasible.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ....................{ IMPORTS                            }....................
from beartype.claw import beartype_this_package

# Enforce type hints across this package with @beartype.
beartype_this_package()

# ....................{ GLOBALS                            }....................
# Declare PEP 8-compliant version constants expected by external automation.

__version__ = '0.0.1'
'''
Human-readable application version as a ``.``-delimited string.

For :pep:`8` compliance, this specifier has the canonical name ``__version__``
rather than that of a typical global (e.g., ``VERSION_STR``).
'''
