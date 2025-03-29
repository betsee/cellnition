# Background

*Cellnition* is an open source simulator to create and analyze Network Finite State Machines (NFSMs)
from regulatory network models.

Regulatory networks, such as Gene Regulatory Networks (GRNs), preside over so many complex phenomena
in biological systems, yet given a specific regulatory network, how do we know what it's capable of doing?

*Cellnition* aims to provide a detailed answer to that question, by treating regulatory
networks as analogue computers, where NFSMs map the sequential logic, 
or analogue "software program",
inherent in the regulatory network's dynamics. 

As an extension and improvement upon attractor landscape analysis,
NFSMs reveal the analogue computing operations inherent in regulatory networks,
allowing for identification of associated "intelligent behaviors".

By capturing the analog programming of regulatory networks, 
NFSMs provide clear identification of:

- Interventions that can induce transitions between stable states (e.g. from "diseased" to "healthy").
- Identification of path-dependencies, representing stable changes occurring after a transient intervention
is applied (e.g. evaluating if a transient treatment with pharmacological agent can permanently heal a condition).
- Identification of inducible cycles of behavior that take the system through a complex
multi-phase process (e.g. wound healing).
- How to target regimes of dynamic activity in a regulatory network (e.g. how to engage a
genetic oscillator instead of a monotonic gene expression in time).

NFSMs have a range of applications, including facilitating the 
identification of strategies to renormalize cancer.

Read more about *Cellnition's* NFSMs in our pre-print publication: [Harnessing the Analogue Computing Power of
Regulatory Networks with the Regulatory Network Machine](https://osf.io/preprints/osf/tb5ys_v1).

Please cite our publication in any work that utilizes *Cellnition*:

```
Pietak, Alexis, and Michael Levin.
 “Harnessing the Analog Computing Power of Regulatory Networks 
 with the Regulatory Network Machine.” OSF Preprints, 2 Dec. 2024. Web.
```
