#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Project paths** (i.e., :class:`pathlib.Path` instances encapsulating
project-specific paths relative to the current directory containing this
project).
'''

# ....................{ IMPORTS                            }....................
from cellnition._util.cache.utilcachecall import callable_cached
from cellnition._util.path.utilpathmake import (
    DirRelative,
    FileRelative,
)
from pathlib import Path

# ....................{ GETTERS ~ package                  }....................
@callable_cached
def get_package_dir() -> Path:
    '''
    :mod:`Path` encapsulating the absolute dirname of the **top-level package**
    (i.e., directory providing this package's top-level package containing at
    least an ``__init__.py`` file) if found *or* raise an exception otherwise.
    '''

    # Path encapsulating the current module.
    MODULE_FILE = Path(__file__)

    # Path encapsulating the current module's package.
    MODULE_PACKAGE_DIR = MODULE_FILE.parent

    # Path encapsulating the dirname of this package's directory relative to the
    # dirname of the subpackage defining the current module.
    PACKAGE_DIR = DirRelative(MODULE_PACKAGE_DIR, '../../')

    # If this package's directory either does not contain our package-specific
    # "error" submodule *OR* does but that path is not a file, raise an
    # exception. This basic sanity check improves the likelihood that this
    # package directory is what we assume it is.
    #
    # Note that we intentionally avoid testing paths *NOT* bundled with release
    # tarballs (e.g., a root ".git/" directory), as doing so would prevent
    # external users and tooling from running tests from release tarballs.
    FileRelative(PACKAGE_DIR, 'error.py')

    # Return this path.
    return PACKAGE_DIR

# ....................{ GETTERS ~ main                     }....................
@callable_cached
def get_main_dir() -> Path:
    '''
    :mod:`Path` encapsulating the absolute dirname of the **root project
    directory** (i.e., directory containing both a ``.git/`` subdirectory and a
    subdirectory providing this project's package) if found *or* raise an
    exception otherwise.
    '''
    # print(f'current module paths: {__package__} [{__file__}]')

    # Path encapsulating the dirname of this project's directory relative to the
    # dirname of the top-level package defining this project.
    MAIN_DIR = DirRelative(get_package_dir(), '../')

    # If this project's directory either does not contain a test-specific
    # subdirectory *OR* does but that path is not a directory, raise an
    # exception. This basic sanity check improves the likelihood that this
    # project directory is what we assume it is.
    #
    # Note that we intentionally avoid testing paths *NOT* bundled with release
    # tarballs (e.g., a root ".git/" directory), as doing so would prevent
    # external users and tooling from running tests from release tarballs.
    DirRelative(MAIN_DIR, 'cellnition')

    # Return this path.
    return MAIN_DIR


@callable_cached
def get_main_readme_file() -> Path:
    '''
    :mod:`Path` encapsulating the absolute filename of the **project readme
    file** (i.e., this project's front-facing ``README.rst`` file) if found
    *or* raise an exception otherwise.

    Note that the :meth:`Path.read_text` method of this object trivially yields
    the decoded plaintext contents of this file as a string.
    '''

    # Perverse pomposity!
    return FileRelative(get_main_dir(), 'README.rst')

# ....................{ GETTERS ~ data : file : csv        }....................
@callable_cached
def get_data_csv_dir() -> Path:
    '''
    :mod:`Path` encapsulating the absolute dirname of the **project-wide
    comma-separated value (CSV) subdirectory** (i.e., directory containing
    ``.csv``-suffixed files describing plaintext data in a columnar format) if
    found *or* raise an exception otherwise.
    '''

    # Perverse pomposity!
    return DirRelative(get_data_dir(), 'csv')

# ....................{ GETTERS ~ data : dir               }....................
@callable_cached
def get_data_dir() -> Path:
    '''
    :mod:`Path` encapsulating the absolute dirname of the **project-wide data
    subdirectory** (i.e., directory providing supplementary non-Python paths
    required throughout this package and thus *not* containing an
    ``__init__.py`` file) if found *or* raise an exception otherwise.
    '''

    # Obverse obviation!
    return DirRelative(get_package_dir(), 'data')


@callable_cached
def get_data_png_glyph_stability_dir() -> Path:
    '''
    :mod:`Path` encapsulating the absolute dirname of the **project-wide
    portable network graphics (PNG) stability glyph subdirectory** (i.e.,
    directory containing ``.png``-suffixed files describing lossless but *not*
    scalable images intended to be embedded in GraphViz-driven visualizations of
    stability networks) if found *or* raise an exception otherwise.
    '''

    # Transverse transcription!
    return DirRelative(get_data_png_dir(), 'glyph_stability')


@callable_cached
def get_data_png_dir() -> Path:
    '''
    :mod:`Path` encapsulating the absolute dirname of the **project-wide
    scalable vector graphics (SVG) subdirectory** (i.e., directory containing
    ``.svg``-suffixed files describing losslessly scalable images) if found *or*
    raise an exception otherwise.
    '''

    # Perverse pomposity!
    return DirRelative(get_data_dir(), 'png')
