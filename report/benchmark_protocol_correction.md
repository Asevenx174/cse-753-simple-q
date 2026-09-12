# Benchmark collection correction

The earlier Acrobot and CartPole runs used random warm-up followed by greedy
training collection because their collectors did not enable exploration noise.
Their paths are preserved in results/analysis/selected_runs_before_fix.csv.
They are preliminary results, not part of the corrected main comparison.

Commit c53b071 enabled collector exploration, explicitly set epsilon to 0.10,
and seeded each training action space. The corrected Acrobot comparison uses
200,000 steps for each algorithm at seeds 0, 1, and 2. Each final policy was
evaluated on the same 100 fresh seeds. Corrected run paths are listed in
results/analysis/selected_benchmarks_corrected.csv.

Corrected final return (mean ± sample SD across three seeds):
DQN: -85.55 ± 5.75; Double DQN: -86.87 ± 3.50. Higher is better.
This small, three-seed comparison does not establish a reliable winner.
A corrected seed-100 Double DQN pilot still fell sharply at the final
checkpoint; its cause remains unproven and the pilot is excluded from the
main table. The NoisyMax and synthetic experiments did not use the affected
benchmark collectors.

Corrected CartPole validation at 50,000 steps, seed 42, gave final
10-episode means of 181.30 (DQN) and 213.40 (Double DQN).
CartPole is supplementary validation, not the main Acrobot result.
