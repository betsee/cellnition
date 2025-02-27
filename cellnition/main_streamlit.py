#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Core application entry-point** (i.e., submodule defining the root
:func:`main` function running this app, intended to be imported from elsewhere
within this codebase at both runtime and test-time).

Specifically, this submodule is imported by:

* The top-level :mod:`cellnition.__main__` submodule, implicitly run by the
  active Python interpreter when passed the ``--m`` option on the command line
  (e.g., ``python3 -m cellnition``).
* Integration tests programmatically exercising app functionality.
'''


# ....................{ KLUDGES ~ path                     }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CAUTION: Kludge PYTHONPATH *BEFORE* importing from this package below.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Explicitly register all files and subdirectories of the parent directory
# containing this module to be importable modules and packages (respectively)
# for the remainder of this Python process if this directory has yet to be
# registered.
#
# Technically, this should *NOT* be required. Streamlit should implicitly
# guarantee this to be the case. Indeed, when Streamlit is run as a module
# (e.g., as "python3 -m streamlit run {app_name}/main.py"), Streamlit does just
# that. Unfortunately, when Streamlit is run as an external command (e.g., as
# "streamlit run {app_name}/main.py"), Streamlit does *NOT* guarantee this to be
# the case. Since Streamlit Cloud runs Streamlit as an external command rather
# than as a module, Streamlit Cloud effectively does *NOT* guarantee this to be
# the case as well.

# Isolate this kludge to a private function for safety.
def _register_dir() -> None:

    # Defer kludge-specific imports. Avert thy eyes, purist Pythonistas!
    from logging import info
    from pathlib import Path
    from sys import path as sys_path

    # Log this detection attempt.
    info('[APP] Detecting whether app package directory requires registration on "sys.path": %s', sys_path)
    # print('Registering app package directory for importation: %s')

    # Path object encapsulating the absolute filename of the file defining the
    # current module. Note that doing so may raise either:
    # * If this file inexplicably does *NOT* exist, "FileNotFoundError".
    # * If this file inexplicably resides under a directory subject to an
    #   infinite symbolic link loop, "RuntimeError".
    main_file = Path(__file__).resolve(strict=True)

    # Absolute dirname of the parent directory containing this app's top-level
    # package, which is guaranteed to be either:
    # * If this app is currently installed editably (e.g., "pip install -e ."),
    #   the repository directory containing the ".git/" directory for this app.
    # * If this app is currently installed non-editably (e.g., "pip install ."),
    #   the equivalent of the "site-packages/" directory for the active Python.
    package_dirname = str(main_file.parents[1])

    # If the current PYTHONPATH does *NOT* already contain this directory...
    if package_dirname not in sys_path:
        # Log this registration attempt.
        info('[APP] Registering app package directory for importation: %s', package_dirname)
        # print('Registering app package directory for importation: %s')

        # Append this directory to the current PYTHONPATH.
        sys_path.append(package_dirname)

# Kludge us up the bomb.
_register_dir()

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CAUTION: Avoid importing anything at module scope *EXCEPT* from official
# Python modules in the standard library guaranteed to exist. Subsequent logic
# in the main() function called below validates third-party runtime
# dependencies of this package to be safely importable, Before performing that
# validation, *NO* other modules are safely importable from.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ....................{ MAIN                               }....................
def main() -> None:
    '''
    Core function running this Streamlit-based web app: **Cellnition.**
    '''

    # ..................{ IMPORTS                            }..................
    import streamlit as st
    from PIL import Image
    import numpy as np
    import pandas as pd

    # ..................{ HEADERS                            }..................
    st.set_page_config(layout="wide") # set a wide page configuration?

    # Human-readable title of this web app.
    st.title('Cellnition')

    #FIXME: Uncomment once we've added a banner logo.
    # banner_image_fn = str(get_data_png_banner_file())
    # banner_image = Image.open(banner_image_fn)
    #
    # st.image(banner_image,
    #          use_column_width='always',
    #          output_format="PNG")

    # App subtitle, if we want it:
    # st.write('Calculating the *slow* changes of bioelectricity')

    # ..................{ LOCALS                             }..................
    #FIXME: Uncomment once we've got parameters.
    # p = ModelParams()  # Create a default parameters instance for model properties
    # sim_p = SimParams() # Create default params for simulation properties
    # l = StringNames() # string labels

    # ..................{ SIDEBAR                            }..................
    # The sidebar will contain all widgets to collect user-data for the
    # simulation. Create and name the sidebar.
    st.sidebar.header('I Am a Header')

    # st.sidebar.write('#### Set simulation variables')
    with st.sidebar:
        pass

# ....................{ MAIN ~ run                         }....................
# Run our Streamlit-based web app.
main()
