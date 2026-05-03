# CF05 CMAN Systolic Trace

## Cycle Table

| Cycle | Input to row 0 | Input to row 1 | PE | Output |
|-------|----------------|----------------|----|--------|
| 0 | A[0,0] = 1 | None | [5, 6; 7, 8] | |
| 1 | A[1,0] | A[0,1] | [15, 6; 19, 8] | |
| 2 | None | A[1,1] | [5, 18; 43, 22] | [19, —; —, —] |
| 3 | None | None | [5, 6; 7, 50] | [19, 22; 43, —] |

## Stats
MACs: $N^3 = 8$ (same as standard matrix multiplication)

Reuses: 6

Memory Accesses:

A: $N^2 = 4$ accesses, once for loading each value.

B: 1, when initially loading the weights.

Output C = $N^2 = 4$, once for writing out each output value.


What is this was an output-stationary array?
If it were output-stationary instead, by definition the partial sums would stay fixes in the PEs, where at the end the contents of the array is the output.
