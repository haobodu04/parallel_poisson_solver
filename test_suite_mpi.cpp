/**
 * @file test_suite_mpi.cpp
 * @brief Test suite for poisson_mpi.cpp
 *
 * This program runs tests for the MPI Poisson solver.
 * The program verifies different cases within tolerances,
 * checks invalid MPI domain decomposition behaviour.
 *
 * The program prints pass/fail and final summary for each test.
 *
 * @author Haobo Du
 * CID: 02372582
 * @date 03/15/2026
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

using namespace std;


/**
 * @brief struct to store coordinates (x, y, z) and solution value u.
 */
struct values {
    double x;
    double y;
    double z;
    double u;
};

/**
 * @brief struct to store full 3D solution
 */
struct solution {
    int Nx;
    int Ny;
    int Nz;
    vector<values> vals;
};

/**
 * @brief store the test metrics
*/
struct metrics{
    double residual = -1.0;
    double error = -1.0;
    double conv_time = -1.0;    // Convergence time
    int    conv_iter = -1;       // Convergence iteration
};

/** 
 * @brief Run the MPI solver and capture its output.
 *
 * @param np     Number of MPI ranks
 * @param args   Command-line arguments for poisson-mpi
 * @param output Captured terminal output
 * @param exit_code Exit status returned 
 * 
 * @return true if output file is captured successfully, false otherwise
 */
 bool run_solver_mpi(int np, const string& args, string& output, int& exit_code){
    const string temp_output = "output_mpi.txt";
    const string cmd = "mpiexec -np " + to_string(np) + " ./poisson-mpi " + args + 
                       " > " + temp_output + " 2>&1";

    exit_code = system(cmd.c_str());

    ifstream in(temp_output.c_str());
    if (!in) {
        output = "";
        remove(temp_output.c_str());
        return false;
    }

    ostringstream ss;
    ss << in.rdbuf();
    output = ss.str();
    in.close();
    remove(temp_output.c_str());
    return true;
 }

 /**
 * @brief Parses solver output for residual, convergence time and index
 *
 * Output: 
 *      Residual: ...
 *      Convergence time: ...
 *      Converged at iteration: ...
 *      Converged at index: ...
 * @param text Full output text produced by solver
 * @param m Parsed metrics
 * @param return true if at lease residual is found, false otherwise
 */
bool output_metrics(const string &text, metrics &m) {
    istringstream in(text);
    string line;
    bool found_residual = false;

    while(getline(in, line)){
        if (line.find("Residual:") == 0){
            istringstream ls(line.substr(9));
            ls >> m.residual;
            if(!ls.fail()){
                found_residual = true;
            }
        } else if (line.find("Convergence time:") == 0){
            istringstream ls(line.substr(17));
            ls >> m.conv_time;
        } else if (line.find("Iterations:") == 0){
            istringstream ls(line.substr(11));
            ls >> m.conv_iter;
        } 
    }
    return found_residual;
}

/**
 * @brief Reads solution from solution.txt
 *
 * This function validates the header and data read.
 *
 * @param name solution file name
 * @param sol output struct containing solution extracted
 * @param reason Error message if reading failed
 * @return true if file is read successfully, false otherwise
 */
bool read_solution(const string &name, solution &sol, string &reason) {
    ifstream in(name.c_str());
    if (!in) {
        reason = "Missing solution.txt";
        return false;
    }

    in >> sol.Nx >> sol.Ny >> sol.Nz;
    if (in.fail()) {
        reason = "Invalid solution header";
        return false;
    }

    const int total = sol.Nx * sol.Ny * sol.Nz;
    sol.vals.clear();
    sol.vals.reserve(total);

    for (int i = 0; i < total; ++i) {
        values a;
        in >> a.x >> a.y >> a.z >> a.u;
        if (in.fail()) {
            reason = "Unexpected end of solution data";
            return false;
        }
        sol.vals.push_back(a);
    }

    return true;
}

/**
 * @brief Returns the exact solution for verification case 1.
 */
double exact_case1(double x, double y, double z) {
    return x * x + y * y + z * z;
}

/**
 * @brief Returns the exact solution for verification case 2.
 */
double exact_case2(double x, double y, double z) {
    return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
}

/**
 * @brief Returns the exact solution for verification case 3.
 */
double exact_case3(double x, double y, double z) {
    return sin(M_PI * x) * sin(4.0 * M_PI * y) * sin(8.0 * M_PI * z);
}

/**
 * @brief Computes the maximum absolute error for verification case 1.
 */
double error_case1(const solution &sol) {
    double max_err = 0.0;
    for (size_t i = 0; i < sol.vals.size(); ++i) {
        const values &a = sol.vals[i];
        const double e = fabs(a.u - exact_case1(a.x, a.y, a.z));
        if (e > max_err) {
            max_err = e;
        }
    }
    return max_err;
}

/**
 * @brief Computes the maximum absolute error for verification case 2.
 */
double error_case2(const solution &sol) {
    double max_err = 0.0;
    for (size_t i = 0; i < sol.vals.size(); ++i) {
        const values &a = sol.vals[i];
        const double e = fabs(a.u - exact_case2(a.x, a.y, a.z));
        if (e > max_err) {
            max_err = e;
        }
    }
    return max_err;
}

/**
 * @brief Computes the maximum absolute error for verification case 3.
 */
double error_case3(const solution &sol) {
    double max_err = 0.0;
    for (size_t i = 0; i < sol.vals.size(); ++i) {
        const values &a = sol.vals[i];
        const double e = fabs(a.u - exact_case3(a.x, a.y, a.z));
        if (e > max_err) {
            max_err = e;
        }
    }
    return max_err;
}


/**
 * @brief Print out the residual, max error and convergence time for each test
 */
void print_metrics(const metrics &m){
    cout << scientific << setprecision(6);

    cout << "       Residual        = " << m.residual << "\n";
    cout << "       Max error       = " << m.error    << "\n";

    if (m.conv_time >= 0){
        cout <<"       Convergence time = " << m.conv_time << "\n";
    } else {
        cout <<"       Convergence time = not found\n";
    }

    if(m.conv_iter >= 0){
        cout <<"       Iterations       = " << m.conv_iter << "\n";
    } else {
        cout <<"       Iterations       = not found\n";
    }

    cout <<defaultfloat;
}

/**
 * @brief Tests MPI verification case 1.
 */
 bool test_case1_mpi(string& reason, metrics& m){
    string output;
    int code = 0;
    solution sol;

    if(!run_solver_mpi(4, "--test 1 --Nx 32 --Ny 32 --Nz 32 --epsilon 1e-9 --Px 2 --Py 2 --Pz 1", output, code)){
        reason = "Failed to captuer solver output";
        return false;
    }

    if (code != 0){
        reason = "Solver failed to run";
        return false;
    }

    if(!output_metrics(output, m)) {
        reason = "Residual not found!";
        return false;
    }

    if(!read_solution("solution.txt", sol, reason)){
        return false;
    }

    if(sol.Nx != 32 || sol.Ny != 32 || sol.Nz != 32){
        reason = "Unexpected grid dimensions";
        return false;
    }

    m.error = error_case1(sol);

    if (m.residual > 1e-8){
        reason = "Residual too large";
        return false;
    }

    if (m.error > 1e-3){
        reason = "Max error too large";
        return false;
    }

    return true;
 }

/**
 * @brief Tests MPI verification case 2.
 */
 bool test_case2_mpi(string& reason, metrics& m){
    string output;
    int code = 0;
    solution sol;

    if(!run_solver_mpi(4, "--test 2 --Nx 64 --Ny 64 --Nz 64 --epsilon 1e-7 --Px 2 --Py 2 --Pz 1", output, code)){
        reason = "Failed to captuer solver output";
        return false;
    }

    if (code != 0){
        reason = "Solver failed to run";
        return false;
    }

    if(!output_metrics(output, m)) {
        reason = "Residual not found!";
        return false;
    }

    if(!read_solution("solution.txt", sol, reason)){
        return false;
    }

    if(sol.Nx != 64 || sol.Ny != 64 || sol.Nz != 64){
        reason = "Unexpected grid dimensions";
        return false;
    }

    m.error = error_case2(sol);

    if (m.residual > 1e-7){
        reason = "Residual too large";
        return false;
    }

    if (m.error > 1e-3){
        reason = "Max error too large";
        return false;
    }

    return true;
 }


 /**
 * @brief Tests MPI verification case 3.
 */
 bool test_case3_mpi(string& reason, metrics& m){
    string output;
    int code = 0;
    solution sol;

    if(!run_solver_mpi(8, "--test 3 --Nx 32 --Ny 64 --Nz 128 --epsilon 1e-7 --Px 2 --Py 2 --Pz 2", output, code)){
        reason = "Failed to captuer solver output";
        return false;
    }

    if (code != 0){
        reason = "Solver failed to run";
        return false;
    }

    if(!output_metrics(output, m)) {
        reason = "Residual not found!";
        return false;
    }

    if(!read_solution("solution.txt", sol, reason)){
        return false;
    }

    if(sol.Nx != 32 || sol.Ny != 64 || sol.Nz != 128){
        reason = "Unexpected grid dimensions";
        return false;
    }

    m.error = error_case3(sol);

    if (m.residual > 1e-7){
        reason = "Residual too large";
        return false;
    }

    if (m.error > 5e-3){
        reason = "Max error too large";
        return false;
    }

    return true;
 }

 /**
  * @brief Test invalid MPI decomposistion.
  */
  bool test_invalid_MPI(string& reason){
    string output;
    int code = 0;

    if (!run_solver_mpi(4, "--test 1 --Nx 32 --Ny 32 --Nz 32 --epsilon 1e-9 --Px 2 --Py 1 --Pz 1", output, code)){
        reason = "Failed to capture solver output";
        return false;
    }

    if (code == 0){
        reason = "Invalid decomposition should fail";
        return false;
    }

    return true;
  }

/**
 * @brief Prints test result and updates pass count.
 */
void report_result(const string &name, bool good, const string &reason, const metrics &m, int &passed) {
    if (good) {
        cout << "[PASS] " << name << "\n";
        print_metrics(m);
        passed++;
    } else {
        cout << "[FAIL] " << name << ": " << reason << "\n";
        print_metrics(m);
    }
}

/**
 * @brief Prints test result for MPI decomposition
 */
void report_MPI(const string &name, bool good, const string reason, int& passed){
    if(good){
        cout << "[PASS] " << name << "\n";
        passed++;
    } else {
        cout << "[FAIL]" << name << ": " << reason << "\n";
    }
}

int main() {
    int passed = 0;
    string reason;
    metrics m;

    reason = "";
    m = metrics{};
    report_result("Verification Case 1", test_case1_mpi(reason, m), reason, m, passed);

    reason = "";
    m = metrics{};
    report_result("Verification Case 2", test_case2_mpi(reason, m), reason, m, passed);

    reason = "";
    m = metrics{};
    report_result("Verification Case 3", test_case3_mpi(reason, m), reason, m, passed);

    reason = "";
    report_MPI("Invalid MPI decomposition", test_invalid_MPI(reason), reason, passed);

    const int total = 4;
    cout << "\nSummary: " << passed << "/" << total << " tests passed\n";
    if (passed == total) {
        return 0;
    } else {
        return 1;
    }
}