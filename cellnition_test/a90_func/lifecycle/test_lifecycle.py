#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2024 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**App lifecycle** (i.e., full process of spinning up, running, and shutting down
this Streamlit-based web app) integration tests.

This submodule integration tests the public API of the
:mod:`cellnition` package.
'''

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: To raise human-readable test errors, avoid importing from
# package-specific submodules at module scope.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from cellnition_test._util.mark.pytskip import skip

# ....................{ TESTS                              }....................
@skip('Streamlit tests currently known to be broken (and irrelevant).')
def test_app_lifecycle(capsys) -> None:
    '''
    Integration test exercising the **maximally trivial app lifecycle** (i.e.,
    workflow spinning up, running, and then immediately shutting down this app).

    Parameters
    ----------
    capsys
        Builtin fixture object permitting pytest-specific capturing (i.e.,
        hiding) of both standard output and error to be temporarily disabled.
    '''

    # Temporarily disable pytest-specific capturing (i.e., hiding) of both
    # standard output and error for the duration of this integration test. Why?
    # Because this test often takes an excessive amount of time and can,
    # moreover, fail to halt in various edge cases outside our control.
    # Capturing output unhelpfully obscures these concerns during local testing.
    with capsys.disabled():
        # Defer test-specific imports.
        #
        # Note that:
        # * Importing "cellnition.main" has the beneficial side effect of running
        #   this Streamlit-based web app to completion.
        # * Attempting to run this app via the "streamlit run" subcommand does
        #   *NOT* halt as expected, as that subcommand understandably runs the
        #   passed app indefinitely.
        from cellnition import main_streamlit

        # Immediately return after doing so.
        return
