.. # ------------------( SEO                                 )------------------
.. # Metadata converted into HTML-specific meta tags parsed by search engines.
.. # Note that:
.. # * The "description" should be no more than 300 characters and ideally no
.. #   more than 150 characters, as search engines may silently truncate this
.. #   description to 150 characters in edge cases.

.. #FIXME: Fill this description in with meaningful content, please.
.. meta::
   :description lang=en:
     Something, something, something.

.. # ------------------( SYNOPSIS                            )------------------

===================
|cellnition-banner|
===================

|ci-badge|

**Cellnition** is an open-source simulator to create and analyze Network Finite
State Machines (NFSMs) from gene regulatory network (GRN) models.

Cellnition is `portably implemented <cellnition codebase_>`__ in Python_,
`continuously stress-tested <cellnition tests_>`__ via `GitHub Actions`_ **×**
tox_ **×** pytest_  **×** Codecov_, and `permissively distributed <cellnition
license_>`__ under the `MIT license`_. For maintainability, cellnition
officially supports *only* the most recently released version of CPython_.

.. # ------------------( TABLE OF CONTENTS                   )------------------
.. # Blank line. By default, Docutils appears to only separate the subsequent
.. # table of contents heading from the prior paragraph by less than a single
.. # blank line, hampering this table's readability and aesthetic comeliness.

|

.. # Table of contents, excluding the above document heading. While the
.. # official reStructuredText documentation suggests that a language-specific
.. # heading will automatically prepend this table, this does *NOT* appear to
.. # be the case. Instead, this heading must be explicitly declared.

.. contents:: **Contents**
   :local:

.. # ------------------( DESCRIPTION                         )------------------

Install
=======

Cellnition is easily installable with _pip, the standard package installer
officially bundled with Python_:

.. code-block:: bash

   pip3 install cellnition

License
=======

Cellnition is `open-source software released <cellnition license_>`__ under the
`permissive MIT license <MIT license_>`__.

.. # ------------------( IMAGES                              )------------------
.. |cellnition-banner| image:: https://github.com/betsee/cellnition/raw/main/cellnition/data/png/cellnition_logo_lion_banner_i.png
   :target: https://cellnition.streamlit.app
   :alt: Cellnition

.. # ------------------( IMAGES ~ badge                      )------------------
.. |app-badge| image:: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
   :target: https://cellnition.streamlit.app
   :alt: Cellnition web app (graciously hosted by Streamlit Cloud)
.. |ci-badge| image:: https://github.com/betsee/cellnition/workflows/test/badge.svg
   :target: https://github.com/betsee/cellnition/actions?workflow=test
   :alt: Cellnition continuous integration (CI) status

.. # ------------------( LINKS ~ cellnition : local          )------------------
.. _cellnition License:
   LICENSE

.. # ------------------( LINKS ~ cellnition : package        )------------------
.. #FIXME: None of these exist, naturally. *sigh*
.. _cellnition Anaconda:
   https://anaconda.org/conda-forge/cellnition
.. _cellnition PyPI:
   https://pypi.org/project/cellnition

.. # ------------------( LINKS ~ cellnition : remote         )------------------
.. _cellnition:
   https://gitlab.com/betsee/cellnition
.. _cellnition app:
   https://cellnition.streamlit.app
.. _cellnition codebase:
   https://gitlab.com/betsee/cellnition
.. _cellnition pulls:
   https://gitlab.com/betsee/cellnition/-/merge_requests
.. _cellnition tests:
   https://gitlab.com/betsee/cellnition/actions?workflow=tests

.. # ------------------( LINKS ~ github                      )------------------
.. _GitHub Actions:
   https://github.com/features/actions

.. # ------------------( LINKS ~ py                          )------------------
.. _Python:
   https://www.python.org
.. _pip:
   https://pip.pypa.io

.. # ------------------( LINKS ~ py : interpreter            )------------------
.. _CPython:
   https://github.com/python/cpython

.. # ------------------( LINKS ~ py : package : test         )------------------
.. _Codecov:
   https://about.codecov.io
.. _pytest:
   https://docs.pytest.org
.. _tox:
   https://tox.readthedocs.io

.. # ------------------( LINKS ~ py : package : web          )------------------
.. _Streamlit:
   https://streamlit.io

.. # ------------------( LINKS ~ py : service                )------------------
.. _Anaconda:
   https://docs.conda.io/en/latest/miniconda.html
.. _PyPI:
   https://pypi.org

.. # ------------------( LINKS ~ soft : license             )------------------
.. _MIT license:
   https://opensource.org/licenses/MIT
