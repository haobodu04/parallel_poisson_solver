# Parallel 3D Poisson Solver

A C++ numerical computing project implementing a three-dimensional Poisson solver with serial, MPI and MPI + CUDA variants. Developed by **Haobo Du** for High Performance Computing coursework at Imperial College London.

The project explores domain decomposition, GPU acceleration and performance profiling. A coursework benchmark reported **145.8 s → 6.93 s (approximately 21×)** for the CUDA implementation relative to the **single-process optimised MPI implementation**, on a **96 × 96 × 96 grid**. See the benchmark qualifications below.

## Implementation

- Second-order finite differences and Jacobi iteration on a structured 3D grid.
- Flattened array storage and a seven-point discrete Laplacian.
- MPI Cartesian domain decomposition with halo exchange using `MPI_Sendrecv`.
- Global residual evaluation using `MPI_Allreduce`.
- Optimised MPI variant checking the residual every 20 iterations, reducing residual computation and collective communication.
- CUDA kernels for Jacobi updates, residual contributions and boundary-face packing/unpacking.
- Thrust device reduction followed by MPI reduction for the global residual.
- Host-staged MPI communication: this implementation does not require CUDA-aware MPI.

## Files

| File | Purpose |
| --- | --- |
| `poisson.cpp` | Serial reference solver |
| `poisson_mpi.cpp` | Baseline MPI solver |
| `mpi_optimised.cpp` | MPI solver with less frequent residual checks |
| `poisson_cuda.cu` | MPI + CUDA solver |
| `test_suite.cpp` | Serial verification harness |
| `test_suite_mpi.cpp` | Baseline MPI verification harness |
| `Makefile` | Build and test targets |
| `job-script.slr` | Original CPU Slurm example |
| `job-script-cu.slr` | Original GPU Slurm example; requires site-specific adaptation |

## Requirements

- A C++17 compiler and GNU Make.
- An MPI implementation exposing `mpic++` and `mpiexec` for MPI builds.
- NVIDIA GPU and CUDA Toolkit, including `nvcc` and Thrust, for the CUDA build.

The supplied CUDA build uses `mpic++` as the host compiler. Compiler/CUDA compatibility and MPI linkage must be checked on the target system.

## Build and run

Run commands from the repository root. Each solver writes `solution.txt`; use separate working directories for simultaneous runs.

### Serial

```bash
make poisson
./poisson --test 2 --Nx 32 --Ny 32 --Nz 32 --epsilon 1e-8
```

### MPI

```bash
make poisson-mpi poisson-mpi-opt
mpiexec -n 4 ./poisson-mpi --test 2 --Nx 32 --Ny 32 --Nz 32 --Px 2 --Py 2 --Pz 1
mpiexec -n 4 ./poisson-mpi-opt --test 2 --Nx 32 --Ny 32 --Nz 32 --Px 2 --Py 2 --Pz 1
```

`Px * Py * Pz` must equal the MPI process count.

### CUDA: single-process starting point

```bash
make poisson-cuda
mpiexec -n 1 ./poisson-cuda --test 2 --Nx 32 --Ny 32 --Nz 32 --Px 1 --Py 1 --Pz 1
```

The CUDA implementation selects a GPU using `world_rank % device_count`. Multi-GPU or multi-node runs require validation against the scheduler's GPU visibility and rank placement. Preserve scheduler-provided GPU visibility unless the cluster's documented launch configuration requires otherwise; the original GPU Slurm script unsets `CUDA_VISIBLE_DEVICES` and should not be copied unchanged to another system.

Use `./poisson --help` for serial options. The MPI/CUDA solvers also accept `--Px`, `--Py` and `--Pz`. Built-in problem cases use `--test 1` through `--test 5`; forcing-file input uses `--forcing` instead of `--test`.

## Verification

```bash
make tests
make tests-mpi
```

The serial harness runs three verification cases. The MPI harness runs three verification cases with four or eight MPI processes, plus an invalid-decomposition check. It invokes `poisson-mpi`; it does **not** automatically test the optimised MPI or CUDA executables.

A documentation-preparation smoke test compiled the serial solver with `g++ -O2 -std=c++17 -Wall -Wextra -pedantic` and ran test 2 on a 12³ grid with a residual tolerance of `1e-8`. It completed in 590 iterations with a reported residual of approximately `9.697e-9`. This is a smoke test, not a full numerical validation. MPI and CUDA were not executed in that environment because their toolchains were unavailable.

## Performance and debugging

### Profiling the initial GPU version

Moving the Jacobi and residual calculations to the GPU initially left runtime at approximately **146.9 s**. The coursework report attributed **137.4 s** to the halo/data-movement stage, while Jacobi updates and residual evaluation took approximately 8.78 s and 0.67 s.

The next optimisation moved boundary-face packing and unpacking into CUDA kernels so that only the required face data crossed the host/device boundary during halo communication. Solution buffers remain on the GPU across iterations and are exchanged by pointer swapping. MPI communication still uses host buffers.

### Reported benchmark

Source: *AERO70011 HPC Coursework Report: Poisson Problem Solver*, Haobo Du, 19 March 2026, Table 4. Global grid: **96³**; single-process comparison.

| Implementation | Runtime | Iterations |
| --- | ---: | ---: |
| Optimised MPI, one CPU process | 145.8 s | 50,521 |
| CUDA, single-process configuration | 6.93 s | 50,521 |

The approximately 21× figure compares these two configurations. It is **not** a speedup against the fastest multi-core MPI run: the report separately records **8.09 s on 48 CPU cores** for the same global grid.

These are historical coursework measurements, not newly reproduced results. The supplied report does not identify the CPU/GPU models or provide repeated-run variability, so the numbers should not be treated as portable performance guarantees. Matching iteration counts alone does not establish numerical equivalence; solution errors and residuals should also be compared.

### CPU strong scaling

Reported optimised MPI results for a fixed 96³ grid:

| CPU cores | Runtime (s) |
| ---: | ---: |
| 1 | 145.8 |
| 2 | 75.3 |
| 4 | 41.04 |
| 8 | 22.9 |
| 16 | 12.9 |
| 32 | 9.84 |
| 48 | 8.09 |

The report's weak-scaling runs also change the number of iterations to convergence. Their total runtimes therefore combine solver-convergence effects with parallel overhead; time per iteration would provide a more useful additional scaling metric.

## Current limitations and next steps

- Add automated comparisons of serial, optimised MPI and CUDA solutions, including refreshed halos when evaluating distributed residuals.
- Record hardware, toolchain versions, process placement, tolerance and repeated timings for reproducible benchmarks.
- Expand CUDA error checking and validate GPU placement on multi-node systems.
- Investigate non-blocking communication and overlap of halo exchange with interior computation.
- Clean up the supplied Makefile's self-referential executable aliases, which may produce circular-dependency warnings. Its `doc` target requires a `Doxyfile` that is not included in the supplied files.

## References

- Mitchell, Schulz and Arnold, *MPI: domain decomposition and halo exchanges*: https://teaching.wence.uk/phys52015/exercises/mpi-stencil/
- NVIDIA Thrust documentation: https://nvidia.github.io/cccl/thrust/
