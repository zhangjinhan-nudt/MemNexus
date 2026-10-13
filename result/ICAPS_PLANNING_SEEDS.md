# Planning seed check

The 20-episode fault-injection suite was executed with seeds `20261010`,
`20261011` and `20261012`. The current templates are deterministic, so all
three runs produce the same mechanism result:

| Seed | Episodes | Recoverable | Stale baseline success | Global replanner success | MemNexus local replanner success | MemNexus constraint violations |
|---:|---:|---:|---:|---:|---:|---:|
| 20261010 | 20 | 16 | 0.00 | 0.80 | 0.80 | 0 |
| 20261011 | 20 | 16 | 0.00 | 0.80 | 0.80 | 0 |
| 20261012 | 20 | 16 | 0.00 | 0.80 | 0.80 | 0 |

The repeated result verifies implementation determinism. It is not a statistical
confidence interval because the current task generator has no stochastic
variation after task construction. The next version should vary deadlines,
resource quantities, alternative branches and fault timing while preserving the
same reproducible seed protocol.
