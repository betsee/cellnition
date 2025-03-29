---
title: Welcome
---
<!-- Hide the title defined above in favour of the banner displayed below while
     still listing this title in the site-wide navigation block to the left.
     Note that this is an obscure MkDocs kludge first publicized here:
     https://github.com/mkdocs/mkdocs/discussions/2431#discussioncomment-7750379
  -->
<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>

![Welcome to Cellnition](https://github.com/user-attachments/assets/50f45c9b-980a-473f-9362-361d3f62061a)

*Cellnition* is an open source simulator to create and analyze Network Finite State Machines (NFSMs)
from regulatory network models.

Regulatory networks, such as Gene Regulatory Networks (GRNs), preside over so many complex phenomena
in biological systems, yet given a specific regulatory network, how do we know what it's capable of doing?

*Cellnition* aims to provide a detailed answer to that question, by treating regulatory
networks as analogue computers, where NFSMs map the sequential logic, 
or analogue "software program",
inherent in the regulatory network's dynamics.

NFSMs have a range of applications, including facilitating the 
identification of strategies to renormalize cancer.

## Installation

*Cellnition* is easily installable with [pip](https://pip.pypa.io), 
the standard package installer
officially bundled with [Python](https://www.python.org):

```bash
pip install cellnition
```

## Issues

Please use the *Cellnition* [issue tracker](https://github.com/betsee/cellnition/issues) to 
report any problems or feedback pertaining to the codebase.  


## About

Read more about *Cellnition's* NFSMs in our pre-print publication: [Harnessing the Analogue Computing Power of
Regulatory Networks with the Regulatory Network Machine](https://osf.io/preprints/osf/tb5ys_v1).

Please cite our publication in any work that utilizes *Cellnition*:

```
Pietak, Alexis, and Michael Levin.
 “Harnessing the Analog Computing Power of Regulatory Networks 
 with the Regulatory Network Machine.” OSF Preprints, 2 Dec. 2024. Web.
```
Cellnition is portably implemented in [Python](),
continuously stress-tested via [GitHub Actions]() **×**
[tox]() **×** [pytest]()  **×** [Codecov](), and [licensed][license] under
a non-commercial use, open source [APACHE license][] with Tufts Open Source License Rider v.1.
For maintainability, cellnition officially supports *only* the most recently released
version of [CPython]().

## Acknowledgements 

*Cellnition* creators are grateful for collaboration opportunities and funding support
from the [Levin Lab]()
at [Tufts University](), with funds for this work supplied from the [Templeton World Charity
Foundation grant TWCF0606](https://www.templetonworldcharity.org/projects-resources/project-database/0606) 
and a sponsored research agreement from [Astonishing Labs](https://astonishinglabs.com/).

## License

*Cellnition* is non-commerical use open source software [licensed][license] under an
[Apache 2.0 license][APACHE license] with Tufts Open Source License
Rider v.1, restricting use to academic purposes only.

[Link References]::
[Levin Lab]: https://as.tufts.edu/biology/levin-lab
[CPython]: https://github.com/python/cpython
[Codecov]: https://about.codecov.io
[pytest]: https://docs.pytest.org
[tox]: https://tox.readthedocs.io
[Python]: https://www.python.org
[Github Actions]: https://github.com/features/actions
[Tufts University]: https://www.tufts.edu
[APACHE license]: https://www.apache.org/licenses/LICENSE-2.0
[license]: https://github.com/betsee/cellnition/blob/main/LICENSE
[Tutorial 1]: https://github.com/betsee/cellnition/blob/main/ipynb/Tutorial1_ContinuousNFSM_v1.ipynb
[Tutorial 2]: https://github.com/betsee/cellnition/blob/main/ipynb/Tutorial2_BooleanNFSM_v1.ipynb