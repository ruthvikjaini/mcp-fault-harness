# Research notes

Researched 29 October 2026 (America/Toronto) before any code was
written. Only the first source was verified on its primary page; the
rest are labelled by how they were checked.

## Primary source

Iannillo, Owotogbe, Natella, Avallone, Kudryavtseva, Kumara,
"Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not
Purchase Fault Tolerance", arXiv:2610.32555, submitted 23 October
2026, accepted at NeurIPS 2026 (poster). Abstract verified on
https://arxiv.org/abs/2610.32555 on 29 October 2026.

What the abstract establishes:

- Current benchmarks assume MCP tools respond correctly and on time;
  agent resilience to realistic tool failures is untested.
- MCP-Injector: a drop-in, protocol-aware fault-injection middleware
  interposed on the MCP tool-call path, with controlled persistence
  (permanent, transient, intermittent) and seedable schedules.
- The fault model is calibrated to the observed incident distribution
  (failure mode x persistence) mined from 57,240 issue and pull-request
  threads across 1,774 open-source MCP repositories.
- Experiment: 1,350 cells crossing two backbone sizes (20B, 120B),
  three agent stacks (custom multi-round plan-and-execute, OpenAI
  Agents SDK, LangChain), and three fault-injection conditions.
- Permanent-heavy stress profile: task-completion score falls 60%,
  from 8.08 to 3.23 out of 10.
- Scale does not purchase fault tolerance for the two gpt-oss
  backbones tested: the 120B retains no larger share of its
  fault-free score (resilience-ratio model, p = 0.96), and its
  completion rate under faults is lower (91.1% vs 95.6%; difference
  4.4 points, 95% CI [-0.2, 9.3]).
- Agent stacks differ in how well they absorb faults, suggesting
  framework-level error handling matters more than model scale.

Secondary coverage (i10X, 27 October 2026, summarising the paper's
tables; not verified against the full text) additionally reports:
most faulted runs still finished, with partial degradation the
typical outcome (56% of runs); 6.0% of faulted runs produced a
plausible answer while barely using tools; and by task type, live-API
tasks fell most (-97%), long sequential chains next (-82%), and short
computational tasks least (-19%). These figures are cited here as
reported, not as verified, and none of them is an input to this
project's simulation.

## Ecosystem context (secondary sources, checked 29 October 2026)

- MCP Canary, "State of MCP Reliability Q4 2026"
  (https://mcpcanary.com/blog/state-of-mcp-reliability-q4-2026/):
  one inspect (connect + `tools/list`) of 30,516 remote endpoints in
  the official registry — 51.4% healthy, 31.2% up behind auth, 10.9%
  broken. Listings last updated in April 2026 or earlier were broken
  40% of the time; October listings 4%.
- HISTOR study, 13 October 2026 (reported by ClawdBytes): of 12,915
  MCP servers installed and started in a sandbox, 41% never answered
  the initialize / tools/list handshake.
- MCP specification, 28 July 2026 release (via Pillar Security's MCP
  guide, updated 11 October 2026): stricter authorization checks;
  session-ID mechanism removed. Used only as background for the
  adapter notes in `docs/ADAPTING_TO_REAL_MCP.md`.

## Alternatives considered and rejected

- PORA, "Learning When More Reasoning Pays Off" (arXiv:2610.27119,
  19 October 2026): effort-allocation learning. Too close to Day 2
  (personalized test-time scaling) and Day 3 (verified-failure-only
  escalation) to be a distinct project.
- Select-TTT (arXiv:2610.36794, 28 October 2026): query-conditioned
  fast-weight updates for in-place test-time training. Genuinely
  current, but an honest CPU reproduction would need a real
  transformer whose MLP down-projections can be repurposed as fast
  weights; a NumPy toy would not have tested the paper's claim.
- Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and
  Cross-Entropy" (arXiv:2610.31642, v2 26 October 2026): two
  floating-point bugs in standard softmax / cross-entropy, with a
  log-space fix. Strong candidate, held for a later night; the
  training-numerics angle is a different project from this one.
- "The One-Pixel of Reasoning" (arXiv:2610.24873, 17 October 2026):
  single-relation cognitive attacks on agents. Interesting, but an
  attack generator is less directly useful for production agent work
  than a resilience harness, and its evaluation needs real models.

## What is simulation, and what is assumed

No code was copied from any paper repository; the implementation is
independent, built from the abstract and the secondary summaries
above. Nothing here measures a real model or a real MCP server.

- "Small" and "large" capabilities are error rates (base misread
  rate 8% vs 3%, guess accuracy 15% vs 20%, fallback mishandle 10%
  vs 5%), not model sizes. The comparison is structural: capability
  and handling vary independently, so resilience ratios can be
  compared the way the paper compares backbones.
- Fault-profile weights are this project's assumption. The paper
  reports a permanent-heavy stress profile, not a mode-by-mode
  breakdown; the weights in `faults.py` / `examples/fault_profile.json`
  encode "permanent-heavy" (60% permanent, 25% intermittent, 15%
  transient) and a plausible mode mix, and are labelled as such.
- A stale payload carries the previous-cycle value, modelled as 15%
  higher than the current value, with an old timestamp. The dataset
  is static, so without this the stale fault would be undetectable
  in value terms and only detectable by timestamp.
- The fallback cache is a snapshot whose values equal the current
  values in this static dataset. Fallback answers get 0.6 credit,
  the penalty for unverified freshness, not for being wrong.
- Latency is virtual (simulated milliseconds, no real sleeping).
  The timeout is 2,000 virtual ms; a `slow` fault on a slow tool can
  exceed it and become a timeout, as in production.
- Servers are in-process simulations of the MCP `tools/list` /
  `tools/call` shape, not real MCP transports. See
  `docs/ADAPTING_TO_REAL_MCP.md` for how the same client logic maps
  onto a real MCP server.
