/** 
 * @file poisson.cpp
 * @brief Serial solver for the 3D poisson equation using Jacobi iteration.
 * 
 * This program solves the Poisson equation on a structured grid using 
 * second-order finite differences. The solver uses the Jacobi iterative
 * method and supports command-line options for selecting test cases,
 * grid sizes and convergence tolerance.
 *
 * The progran outputs the converged numerical solution to solution.txt.
 *
 * @author Haobo Du
 * CID: 02372582
 * @date 03/07/2026
*/
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

/**
 * @brief Store command-line options for later mroe convenient use
*/
struct Options {
    bool help       = false;     // if help is required by user input
    bool if_test    = false;     // if --test is the input
    bool if_forcing = false;     // if --forcing is the input

    int  test       = 1;         // test number commanded among five tests
    int  Nx         = -1;        // number of grid points in x
    int  Ny         = -1;        // number of grid points in y
    int  Nz         = -1;        // number of grid points in z

    bool Nx_set     = false;     // whether Nx is set by user
    bool Ny_set     = false;     // whether Ny is set by user
    bool Nz_set     = false;     // whether Nz is set by user

    double epsilon  = 1e-8;      // residual convergence threshold
    string forcing_file;         // the forcing file name
};

/**
 * @brief Print command-line options
 * 
 * Display the list of available command-line options supported by the 
 * Poisson solver.  If the user keys in "--help", this function is called.
*/
void help() {
    cout << "Allowed options: \n" 
         << "--help             Print available options.\n"
         << "--forcing arg      Input forcing file\n"
         << "--test arg (=1)    Test case to use (1-5)\n"
         << "--Nx arg (=32)     Number of grid points (x)\n"
         << "--Ny arg (=32)     NUmber of grid points (y)\n"
         << "--Nz arg (=32)     Number of grid points (z)\n"
         << "--epsilon (=1e-08) Residual threshold";
}

/**
 * @brief convert 3D grid to 1D array index
 * 
 * The grid is converted to a 1D array storage to save memory efficiency. 
 * Example of how this works:
 * For Nx = 3, Ny = 4, Nz = 5
 * For each x level, there are Ny*Nz = 20 elements, at (2, 1, 3)
 * index become to index = 2*(20) + 1*5 + 3 = 48
 * 
 * Since indices is used frequently throughout this coursework, the conversion
 * make it more efficient.
*/
inline int index(int i, int j, int k, int Ny, int Nz){
    return i * Ny * Nz + j * Nz + k;
}


/**
 *@brief Utilise Option strucutre to give responses to command-line
 * 
 * @param argc Number of command-line arguments.
 * @param argv Number of command-line argument strings.
 * @return Output Options struct.
 *
 * @throws runtime_error if input is missing or invalid
*/
Options respond(int argc, char* argv[]){
    Options options;
    
    for (int i = 1; i < argc; ++i){
        string arg = argv[i];

        if (arg == "--help"){
            options.help = true;
        }else if (arg == "--test"){
            if (i + 1 >= argc){
                throw runtime_error("Hey! Missing number after --test!");
            }
            options.if_test = true;
            options.test = stoi(argv[++i]);
        } else if(arg == "--forcing"){
            if (i + 1 >= argc){
                throw runtime_error("Hey! Missing file name after --forcing!");
            }
            options.if_forcing = true;
            options.forcing_file = argv[++i];
        } else if(arg == "--Nx"){
            if (i + 1 >= argc) {
                throw runtime_error("Well...Missing number after --Nx!");
            }
            options.Nx_set = true;
            options.Nx = stoi(argv[++i]);
        } else if(arg == "--Ny"){
            if (i + 1 >= argc) {
                throw runtime_error("Ummm! Missing number after --Ny!");
            }
            options.Ny_set = true;
            options.Ny = stoi(argv[++i]);
        } else if(arg == "--Nz"){
            if (i + 1 >= argc) {
                throw runtime_error("Uh-oh! Missing number after --Nz!");
            }
            options.Nz_set = true;
            options.Nz = stoi(argv[++i]);
        } else if(arg == "--epsilon"){
            if(i + 1 >= argc){
                throw runtime_error("Ooops! Missing number after --epsilon!");
            }
            options.epsilon = stod(argv[++i]);
        } else {
            throw runtime_error("Your input is not on our menu, could you please check it again? Your input: " + arg);
        }
    }
    
    if (options.help){
        return options;
    }

    if (options.if_test && options.if_forcing) {
        throw runtime_error("Use either --test or --forcing, not both!");
    }

    if (!options.if_test && !options.if_forcing){
        throw runtime_error("Provide one of --test or --forcing!");
    }

    if (options.if_forcing && (options.Nx_set || options.Ny_set || options.Nz_set)) {
        throw runtime_error("--Nx/--Ny/--Nz are only allowed with --test!");
    }

    if (options.if_test && (options.test < 1 || options.test > 5)){
        throw runtime_error("--test must be in range 1-5");
    }

    return options;
}

/**
 * @brief Set default grid dimentions for each test
 *
 * In case the user input doesn't provide --test and --forcing, this function assures that
 * the code still gives sensible result with default setting.
 *
 * Default for each test :
 * - Test 1: 32 x 32 x 32
 * - Test 2: 64 x 64 x 64
 * - Test 3: 32 x 64 x 128
 * - Test 4: 64 x 64 x 64
 * - Test 5: 64 x 64 x 64
 * 
 * @param test Test case number (1-5)
 * @param Nx Number of grid points in x.
 * @param Ny Number of grid points in y.
 * @param Nz Number of grid points in z.
 *
 * @throws runtime_error if test is not 1-5.
 */
void set_default(int test, int& Nx, int& Ny, int& Nz){
    if (test == 1){
        Nx = Ny = Nz = 32;
    } else if(test == 2){
        Nx = Ny = Nz = 64;
    } else if(test == 3){
        Nx = 32;
        Ny = 64;
        Nz = 128;
    } else if(test == 4 || test == 5){
        Nx = Ny = Nz = 64;
    } else {
        throw runtime_error("Invalid test case, should be 1-5.");
    }
}


/**
 * @brief Set up forcing values and boundary conditions for the selected test case.
 * 
 * Initialise the f(x, y, z) over full grid and applies Dirichlet boundary values for test 1 - 3
 * (verifications). For tests 4 and 5, boundary condition of u = 0 are used. Interior values are 
 * set to 0 for Jacobi iteration.
 *
 * @param test x Test case number (1-5)
 * @param u Solution array at current iteration.
 * @param u_jac Solution array for the used of Jacobi.
 * @param f forcing array.
 * @param Nx Number of points in x.
 * @param Ny Number of points in y.
 * @param Nz Number of points in z.
 * @param hx Grid spacing in x.
 * @param hy Grid spacing in y.
 * @param hz Grid spacing in z.
*/
void test_case(int test, vector<double>& u, vector<double>& u_jac, 
                   vector<double>& f, int Nx, int Ny, int Nz, double hx, double hy, double hz){
    for (int i = 0; i < Nx; ++i){
        for (int j = 0; j < Ny; ++j){
            for (int k = 0; k < Nz; ++k){
                const double xi = i * hx;
                const double yj = j * hy;
                const double zk = k * hz;
                const int idx = index(i, j, k, Ny, Nz);

                if (test == 1){
                    f[idx] = 6.0;
                } else if (test == 2){
                    f[idx] = -3 * M_PI * M_PI * sin(M_PI * xi) * sin(M_PI * yj) * sin(M_PI * zk);
                } else if (test == 3){
                    f[idx] = -81.0 * M_PI * M_PI * sin(M_PI * xi) * sin(4 * M_PI * yj) * sin(8 * M_PI * zk);
                } else if (test == 4){
                    const double dx = xi - 0.5;
                    const double dy = yj - 0.5;
                    const double dz = zk - 0.5;
                    f[idx] = 100.0 * exp(-100.0 * (dx * dx + dy * dy + dz * dz));
                } else if (test == 5){
                    if (xi < 0.5){
                        f[idx] = 1.0;
                    } else if (xi >= 0.5){
                        f[idx] = -1.0;
                    }
                }
                
                if (i == 0 || i == Nx -1|| 
                    j == 0|| j == Ny - 1|| k == 0|| k == Nz -1){
                        double bc = 0.0;
                        if (test == 1){
                            bc = xi * xi + yj * yj + zk * zk;
                        }else if (test == 2){
                            bc = sin(M_PI * xi) * sin(M_PI * yj) * sin(M_PI * zk);
                        }else if (test == 3){
                            bc = sin(M_PI * xi) * sin(4.0 * M_PI * yj) * sin(8.0 * M_PI * zk);
                        }
                        u[idx] = bc;
                        u_jac[idx] = bc;
                    } else {
                        u[idx] = 0.0;
                        u_jac[idx] = 0.0;
                    }
            }
        }
    }
}


/**
 * @brief Read forcing data from file and initialise solver arrays.
 *
 * Expected file format: 
 *          First line: Nx Ny Nz
 *          Following:  x y z value
 * This function reads grid sizes and compute grid spacings as required in the handout,
 * stores the input value into f at each points and applies homogeneous Dirichlet boundary 
 * initialization (u = 0) on boundary.
 *
 * @param name Input forcing file name.
 * @param u Solution array.
 * @param u_next Solution array for Jacobi updates.
 * @param f Forcing array.
 * @param Nx Number of grid points in x (read from file).
 * @param Ny Number of grid points in y (read from file).
 * @param Nz Number of grid points in z (read from file).
 * @param hx Grid spacing in x (computed).
 * @param hy Grid spacing in y (computed).
 * @param hz Grid spacing in z (computed).
*/
void read_forcing_file(const string& name, vector<double>& u, vector<double>& u_jac, vector<double>& f, 
                       int& Nx, int& Ny, int& Nz, double& hx, double& hy, double& hz){
    ifstream in(name);
    if (!in){
        throw runtime_error("Failed to open file: " + name + "!");
    }

    in >> Nx >> Ny >> Nz;
    if (!in){
        throw runtime_error("Invalid forcing file header! Expected: Nx Ny Nz");
    }
    if (Nx < 2 || Ny < 2 || Nz < 2){
        throw runtime_error("Forcing file grid must satisfy Nx, Ny, Nz >= 2");
    }

    hx = 1.0 / (Nx - 1);
    hy = 1.0 / (Ny - 1);
    hz = 1.0 / (Nz - 1);

    u = vector<double>(Nx * Ny * Nz, 0.0);
    u_jac = vector<double>(Nx * Ny * Nz, 0.0);
    f = vector<double>(Nx * Ny * Nz, 0.0);

    const int total = Nx * Ny * Nz;

    for (int n = 0; n < total; ++n){
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double value = 0.0;
        in >> x >> y >> z >> value;
        if (!in){
            throw runtime_error("Forcing file ended early. Expected Nx * Ny * Nz data lines!");
        }

        const int i = n % Nx;
        const int j = (n / Nx) % Ny;
        const int k = n / (Nx * Ny);
        const int idx = index(i, j, k, Ny, Nz);

        f[idx] = value;

        if (i == 0 || i == Nx -1|| 
                    j == 0|| j == Ny - 1|| k == 0|| k == Nz -1){
            u[idx] = 0.0;
            u_jac[idx] = 0.0;
        }
    }
}



/**
 * @brief Compute the norm
 * 
 * residual is defined as r = f - Laplacian
 *
 * @param u solution array
 * @param f forcing array
 * @param Nx Number of grid points in x
 * @param Ny Number of grid points in y
 * @param Nz Number of grid poitns in z
 * @param hx grid spacing in x
 * @param hy grid spacing in y
 * @param hz grid spacing in z
 * @return residual norm
*/
double Residual(const vector<double>& f, const vector<double>& u, int Nx, int Ny, int Nz, 
                double hx, double hy, double hz){
    const double denom_x = 1.0 / (hx * hx);
    const double denom_y = 1.0 / (hy * hy);
    const double denom_z = 1.0 / (hz * hz);

    double sum = 0.0;

    for (int i = 1; i < Nx - 1; ++i){
        for (int j = 1; j < Ny - 1; ++j){
            for (int k = 1; k < Nz - 1; ++k){
                int idx = index(i, j, k, Ny, Nz);

                double lapX = (u[index(i+1, j, k, Ny, Nz)] - 2.0 * u[idx] + u[index(i - 1, j, k, Ny, Nz)]) * denom_x;
                double lapY = (u[index(i, j+1, k, Ny, Nz)] - 2.0 * u[idx] + u[index(i, j - 1, k, Ny, Nz)]) * denom_y;
                double lapZ = (u[index(i, j, k+1, Ny, Nz)] - 2.0 * u[idx] + u[index(i, j, k - 1, Ny, Nz)]) * denom_z;

                double lap = lapX + lapY + lapZ;
                double r = f[idx] - lap;
                sum += r*r;
            }
        }
    }
    return sqrt(sum);
}

/**
 * @brief Apply the Jacobi Iterative Method as shown in handout
 *        
 * Updates each interior points using Jacobi while keep boundary not modified.
 * 
 * As required: f(x, y, z) = 6;
 * 
 * Other values are set to 0 at this stage.
 *
 * @param u solution array
 * @param u_jac array used by Jacobi
 * @param f forcing array
 * @param Nx Number of grid points in x
 * @param Ny Number of grid points in y
 * @param Nz Number of grid poitns in z
 * @param hx grid spacing in x
 * @param hy grid spacing in y
 * @param hz grid spacing in z
*/
void jacob(const vector<double>& u, vector<double>& u_jac, vector<double>& f, int Nx, int Ny, int Nz,
           double hx, double hy, double hz){
    const double denom_x2 = 1.0 / (hx * hx);
    const double denom_y2 = 1.0 / (hy * hy);
    const double denom_z2 = 1.0 / (hz * hz);
    const double denom = 2.0 * (denom_x2 + denom_y2 + denom_z2);

    for (int i = 1; i < Nx - 1; ++i){
        for (int j = 1; j < Ny - 1; ++j){
            for (int k = 1; k < Nz - 1; ++k){
                int idx = index(i, j, k, Ny, Nz);

                u_jac[idx] = ((u[index(i + 1, j, k, Ny, Nz)] + u[index(i - 1, j, k, Ny, Nz)]) * denom_x2 + 
                              (u[index(i, j + 1, k, Ny, Nz)] + u[index(i, j - 1, k, Ny, Nz)]) * denom_y2 + 
                              (u[index(i, j, k + 1, Ny, Nz)] + u[index(i, j, k - 1, Ny, Nz)]) * denom_z2 - 
                               f[idx]) / denom;

            }
        }
    }
}

/**
 * @brief Write the converged values of u(x, y, z) to solution.txt+
 */

void write_solution(const string& filename, const vector<double>& u, 
                    int Nx, int Ny, int Nz, double hx, double hy, double hz){
    ofstream file(filename);
    if(!file) {
        throw runtime_error("Failed to open output file: " + filename);
    }

    file << Nx << " " << Ny << " " << Nz << "\n";
    file << scientific << setprecision(10);

    for (int k = 0; k < Nz; ++k){
        for (int j = 0; j < Ny; ++j){
            for (int i = 0; i < Nx; ++i){
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                double val = u[index(i, j, k, Ny, Nz)];

                file << setw(18) << x 
                     << setw(18) << y 
                     << setw(18) << z 
                     << setw(22) << val << "\n";
            }
        }
    }
}

int main(int argc, char* argv[]){
    try { 
        const Options opt = respond(argc, argv);

        if (opt.help) { 
            help();
            return 0;
        }
        int Nx = opt.Nx;
        int Ny = opt.Ny;
        int Nz = opt.Nz;

        if (opt.if_test){
            int dNx = 0;
            int dNy = 0;
            int dNz = 0;
            set_default(opt.test, dNx, dNy, dNz);
            if (opt.Nx_set){
                Nx = opt.Nx;
            }else{
                Nx = dNx;
            }
            if (opt.Ny_set){
                Ny = opt.Ny;
            }else{
                Ny = dNy;
            }
            if (opt.Nz_set){
                Nz = opt.Nz;
            }else{
                Nz = dNz;
            }
        }

        double hx = 0.0;
        double hy = 0.0;
        double hz = 0.0;

        vector<double> u;
        vector<double> u_jac;
        vector<double> f;

        if (opt.if_test){
            if (Nx < 2 || Ny < 2 || Nz < 2){
                throw runtime_error("Nx, Ny and Nz must each be at least 2");
            }
            hx = 1.0/(Nx - 1);
            hy = 1.0/(Ny - 1);
            hz = 1.0/(Nz - 1);

            u = vector<double> (Nx * Ny * Nz, 0.0);
            u_jac = vector<double> (Nx * Ny * Nz, 0.0);
            f = vector<double> (Nx * Ny * Nz, 0.0);
            test_case(opt.test, u, u_jac, f, Nx, Ny, Nz, hx, hy, hz);
        } else {
            read_forcing_file(opt.forcing_file, u, u_jac, f, Nx, Ny, Nz, hx, hy, hz);
        }

        double residual_l2 = Residual(f, u, Nx, Ny, Nz, hx, hy, hz);
        const int max_iter = 1000000;
        int iter = 0;

        double epsilon = opt.epsilon;

        auto start = chrono::high_resolution_clock::now(); //start timer for the test

        while (residual_l2 > epsilon && iter < max_iter){
            jacob(u, u_jac, f, Nx, Ny, Nz, hx, hy, hz);
            u.swap(u_jac);
            residual_l2 = Residual(f, u, Nx, Ny, Nz, hx, hy, hz);
            ++iter;
        }

        if (iter == max_iter && residual_l2 > epsilon){
            throw runtime_error("Jacobi iteration did not converge within max interations!");
        }

        auto end = chrono::high_resolution_clock::now(); //stop timer
        double time = chrono::duration<double>(end - start).count(); // store the duration of test

        write_solution("solution.txt", u, Nx, Ny, Nz, hx, hy, hz);

        cout << scientific << setprecision(10);
        cout << "Residual: " << setprecision(10) << residual_l2 <<"\n";
        cout << "Convergence time: " << time << "\n";
        cout << "Iterations: " << iter << "\n";

        return 0;
    }
    catch (const exception& e){
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}