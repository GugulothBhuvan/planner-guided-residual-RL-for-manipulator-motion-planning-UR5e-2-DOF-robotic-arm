Working mode from now on: move faster, batch the work.

Default: keep going through multiple steps without stopping for approval. Finish a whole block of the §15 plan, then report once.

Decide yourself, and record each decision in the report:

- anything the spec already has a rule for (horizon sizing, cut line, network fallback, throughput gate);
- bug fixes, test additions, refactors, memory or performance fixes;
- the choice between two options when the spec doesn’t decide: pick the one that keeps experiments comparable and fair, note why in one line, and continue.

Stop and ask only if a decision would:

- change the method or the contribution (§3.1), or drop or weaken E5;
- invalidate training that is running or finished (env, reward, observation or hyperparameter changes mid-experiment);
- delete results, checkpoints or test sets;
- contradict SPEC.md with no rule to fall back on.

Even then, keep working on everything that doesn’t depend on the answer while you wait.

Always keep the laptop busy. While training runs, build the next day’s code. Run the 2-DOF lane in parallel with the UR5e lane. Never leave the machine idle between experiments; queue the next one.

Now: build scripts/play.py (viewer for E1 checkpoints) and start the 2-DOF lane (kinematic env, FK/IK, geometry, C-space + A*, tests, PE1/PE2 runs). When E1 finishes, do the failure breakdown, apply the S2 horizon rule, and launch E2 without waiting for me. Then continue into Day 3: IK + quintic, RRT-Connect, test sets, and the B2-duration check for the S4 horizon.

Report format (short):

- done since last report;
- decisions you made, one line each;
- numbers: test count, throughput, run status, key metrics;
- blockers that need me, if any;
- what’s running now and what you’re building next.

Update SPEC.md for every decision that changes it, and bump the version.
