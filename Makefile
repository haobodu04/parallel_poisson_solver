CXX := g++
MPICXX := mpic++
NVCC := nvcc

CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
MPIFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
NVCCFLAGS := -O2 -std=c++17 --allow-unsupported-compiler

SERIAL_SRC := poisson.cpp
SERIAL_BIN := poisson

SERIAL_TEST_SRC := test_suite.cpp
SERIAL_TEST_BIN := test_suite

MPI_SRC := poisson_mpi.cpp
MPI_BIN := poisson-mpi

MPI_OPT_SRC := mpi_optimised.cpp
MPI_OPT_BIN := poisson-mpi-opt

CUDA_SRC := poisson_cuda.cu
CUDA_BIN := poisson-cuda

MPI_TEST_SRC := test_suite_mpi.cpp
MPI_TEST_BIN := test_suite_mpi

DOC_CONFIG := Doxyfile

.PHONY: all poisson poisson-mpi poisson-mpi-opt poisson-cuda tests tests-mpi run run-mpi run-mpi-opt run-cuda doc clean

all: poisson poisson-mpi poisson-mpi-opt poisson-cuda

poisson: $(SERIAL_BIN)

poisson-mpi: $(MPI_BIN)

poisson-mpi-opt: $(MPI_OPT_BIN)

poisson-cuda: $(CUDA_BIN)

$(SERIAL_BIN): $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(SERIAL_TEST_BIN): $(SERIAL_TEST_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(MPI_BIN): $(MPI_SRC)
	$(MPICXX) $(MPIFLAGS) $< -o $@

$(MPI_OPT_BIN): $(MPI_OPT_SRC)
	$(MPICXX) $(MPIFLAGS) $< -o $@

$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) $< -o $@

$(MPI_TEST_BIN): $(MPI_TEST_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

tests: $(SERIAL_BIN) $(SERIAL_TEST_BIN)
	./$(SERIAL_TEST_BIN)

tests-mpi: $(MPI_BIN) $(MPI_TEST_BIN)
	./$(MPI_TEST_BIN)

run: $(SERIAL_BIN)
	./$(SERIAL_BIN) --help

run-mpi: $(MPI_BIN)
	mpiexec -n 1 ./$(MPI_BIN) --help --Px 1 --Py 1 --Pz 1

run-mpi-opt: $(MPI_OPT_BIN)
	mpiexec -n 1 ./$(MPI_OPT_BIN) --help --Px 1 --Py 1 --Pz 1

run-cuda: $(CUDA_BIN)
	mpiexec -n 1 ./$(CUDA_BIN) --help --Px 1 --Py 1 --Pz 1

doc:
	doxygen $(DOC_CONFIG)

clean:
	rm -f $(SERIAL_BIN) $(SERIAL_TEST_BIN) \
	      $(MPI_BIN) $(MPI_OPT_BIN) $(CUDA_BIN) $(MPI_TEST_BIN) \
	      solution.txt solver_output.txt solver_output_mpi.txt