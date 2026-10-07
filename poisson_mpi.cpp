/**
 * @file poisson_mpi.cpp
 * @brief MPI solver for the 3D Poisson equation using Jacobi iteration.
 * 
 * This program solves the Poisson equation on a structured grid using
 * second-order finite differences and Jacobi iteration, parallelised with MPI
 * using a 3D Cartesian domain decomposition.
 *
 * The program supports command-line interface as the serial version, 
 * with addtional options --Px, --Py and --Py for the MPI process grid.
 * 
 * Input and output remain single files, as required by the instruction.
 * @author Haobo Du
 * CID: 02372582
 * @date 03/12/2026
 */

 #include <mpi.h>

 #include <iostream>
 #include <iomanip>
 #include <sstream>
 #include <string>
 #include <vector>
 #include <stdexcept>
 #include <iostream>
 #include <fstream>
 #include <exception>
 #include <cmath>
 #include <chrono>
 #include <algorithm>

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

    int  Px         = 1;
    int  Py         = 1;
    int  Pz         = 1;
    bool Px_set     = false;
    bool Py_set     = false;
    bool Pz_set     = false;
};

/**
 * @brief Store local domain information for each sub-block
 */
 struct LocalDomain {
    // global start index
    int sx = 0;
    int sy = 0;
    int sz = 0;

    // local size of sub-block
    int nx = 0;
    int ny = 0;
    int nz = 0;
 };

/**
 * @brief Print command-line options
 * 
 * Display the list of available command-line options supported by the 
 * Poisson solver.  If the user keys in "--help", this function is called.
*/
void help(){
    cout << "Allowed options: \n" 
         << "--help             Print available options.\n"
         << "--forcing arg      Input forcing file\n"
         << "--test arg (=1)    Test case to use (1-5)\n"
         << "--Nx arg (=32)     Number of grid points (x)\n"
         << "--Ny arg (=32)     NUmber of grid points (y)\n"
         << "--Nz arg (=32)     Number of grid points (z)\n"
         << "--epsilon (=1e-08) Residual threshold\n"
         << "--Px arg (=1)      Number of processes (x)\n"
         << "--Py arg (=1)      Number of processes (y)\n"
         << "--Pz arg (=1)      Number of processes (z)\n";
}

/**
 * @brief convert 3D grid to 1D array index for global index
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
inline int global_index(int i, int j, int k, int Ny, int Nz){
    return i * Ny * Nz + j * Nz + k;
}

/**
 * @brief Local linear index conversion for sub-blocks
 *
 * Follow the same methodology but +2 to leave space for Laplacian
 */
inline int local_index(int i, int j, int k, int ny, int nz){
    return (i * (ny + 2) + j) * (nz + 2) + k;
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
        } else if (arg == "--Px"){
            if (i + 1 >= argc) {
                throw runtime_error("Woah! Missing number after --Px");
            }
            options.Px = stoi(argv[++i]);
            options.Px_set = true;
        } else if (arg == "--Py"){
            if (i + 1 >= argc){
                throw runtime_error("Ooops! Missing number after --Py");
            }
            options.Py = stoi(argv[++i]);
            options.Py_set = true;
        } else if (arg == "--Pz"){
            if (i + 1 >= argc){
                throw runtime_error("Oh...Missing number after --Pz");
            }
            options.Pz = stoi(argv[++i]);
            options.Pz_set = true;
        }
        else {
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

    if (options.Px < 1 || options.Py < 1 || options.Pz < 1){
        throw runtime_error("--Px, --Py and --Pz must each be at least 1");
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
 * @brief Compute number of grid points assigned to a process in 1D.
 * 
 * This function allocate the number of the points evenly to each 
 * MPI process and also arrange the remainder if exists.
 *
 * @param N Total number of grid points in the global dimension
 * @param P Number of MPI processes
 * @param coord Coordinate of "current" MPI process
 *
 * @return Number of grid points owned by this process
 */
 int local_size_1D(int N, int P, int coord){
    const int base = N / P;
    const int rem  = N % P;
    if (coord < rem){
        return base + 1;
    } else {
        return base;
    }
 }

 /**
  * @brief Return global start index of a process in 1D
  * 
  * After the grid points are distributed using local_size_1D(), each MPI process
  * owns a consecutive block of the global grid. This function computes the global
  * starting index of that block.
  *
  * @param N Total number of grid points in the global dimension
  * @param P Number of MPI processes
  * @param coord Coordinate of "current" MPI process
  *
  * @return Global starting index of this process in corresponding dimension
  */
int local_start_1D(int N, int P, int coord){
    int base = N / P;
    int rem  = N % P;

    return coord * base + min(coord, rem);
}

/**
 * @brief Construct the local subdomain owned by an MPI process.
 *
 * In the MPI solver, the global grid (Nx x Ny x Nz) is decomposed across 
 * a 3D Cartesian grid of processes (Px x Py x Pz). This function determines 
 * 1) the starting global index of the block (sx, sy, sz); 2) the number of grid
 * points owned in each direction (nx, ny, nz) using local_size_1D() and 
 * using_local_start_1D().
 *
 * @param Nx, Ny, Nz Global grid dimensions.
 * @param Px, Py, Pz Number of MPI processes in each dimension
 * @param cx, cy, cz Cartesian coordinates of the current MPI process
 * 
 * @return LocalDomain structure for this process
 */
 LocalDomain build_local_domain(int Nx, int Ny, int Nz, int Px, int Py, int Pz,
                               int cx, int cy, int cz){
    LocalDomain D;
    D.sx = local_start_1D(Nx, Px, cx);
    D.sy = local_start_1D(Ny, Py, cy);
    D.sz = local_start_1D(Nz, Pz, cz);

    D.nx = local_size_1D(Nx, Px, cx);
    D.ny = local_size_1D(Ny, Py, cy);
    D.nz = local_size_1D(Nz, Pz, cz);

    return D;
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
 * @brief Initialise local solution and forcing arrays for built-in test cases.
 * 
 * Each MPI process sets the values for its owned subdomain based on the
 * selected test case. Forcing terms are computed using the analytical
 * expressions, and Dirichlet boundary conditions are applied on global
 * domain boundaries.
 *
 * @param test x Test case number (1-5)
 * @param u Solution array at current iteration.
 * @param u_jac Solution array for the used of Jacobi.
 * @param f forcing array.
 * @param dom LocalDomain structure describing the process domain
 * @param Nx Number of points in x.
 * @param Ny Number of points in y.
 * @param Nz Number of points in z.
 * @param hx Grid spacing in x.
 * @param hy Grid spacing in y.
 * @param hz Grid spacing in z.
*/
void test_case_local(int test, vector<double>& u, vector<double>& u_jac, 
                     vector<double>& f, const LocalDomain& dom, 
                     int Nx, int Ny, int Nz, double hx, double hy, double hz){
    for (int i = 1; i <= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            for (int k = 1; k <= dom.nz; ++k){
                const int gi = dom.sx + (i - 1);  // global index
                const int gj = dom.sy + (j - 1);
                const int gk = dom.sz + (k - 1);

                const double x = gi * hx;
                const double y = gj * hy;
                const double z = gk * hz;

                const int idx = local_index(i, j, k, dom.ny, dom.nz);

                if (test == 1){
                    f[idx] = 6.0;
                } else if (test == 2){
                    f[idx] = -3.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
                } else if (test == 3){
                    f[idx] = -81.0 * M_PI * M_PI * sin(M_PI * x) * sin(4.0 * M_PI * y) * sin(8.0 * M_PI * z);
                } else if (test == 4){
                    const double dx = x - 0.5;
                    const double dy = y - 0.5;
                    const double dz = z - 0.5;
                    f[idx] = 100.0 * exp(-100.0 * (dx * dx + dy * dy + dz * dz));
                } else if (test == 5){
                    if (x < 0.5){
                        f[idx] = 1.0;
                    } else {
                        f[idx] = -1.0;
                    }
                }

                if (gi == 0 || gi == Nx - 1 ||gj == 0 || gj == Ny - 1 || gk == 0 || gk == Nz - 1){
                    double bc = 0.0;
                    if (test == 1) {
                        bc = exact_case1(x, y, z);
                    } else if (test == 2) {
                        bc = exact_case2(x, y, z);
                    } else if (test == 3) {
                        bc = exact_case3(x, y, z);
                    }
                    u[idx]     = bc;
                    u_jac[idx] = bc;
                } else {
                    u[idx]     = 0.0;
                    u_jac[idx] = 0.0;
                }
            }
        }
    }
}

/**
 * @brief Read forcing file data from file on the root (rank 0) MPI process.
 * 
 * The root process reads the forcing file containing the global grid dimensions
 * and forcing values for the entire domain. The data is stored in a global
 * forcing array, which will later be distributed to each MPI process.
 */
void read_forcing_file(const string& name, vector<double>& global_f, 
                       int& Nx, int& Ny, int& Nz, double& hx, 
                       double& hy, double& hz){
    ifstream in(name);
    if(!in){
        throw runtime_error("Failed to open forcing file: " + name);
    }

    in >> Nx >> Ny >> Nz;
    if(!in){
        throw runtime_error("Invalid forcing file header");
    }
    if (Nx < 2 || Ny < 2 || Nz < 2){
        throw runtime_error("Forcing file grid must satisfy Nx, Ny, Nz >= 2");
    }

    hx = 1.0 / (Nx - 1);
    hy = 1.0 / (Ny - 1);
    hz = 1.0 / (Nz - 1);

    global_f = vector<double>(Nx * Ny * Nz, 0);

    for (int n = 0; n < Nx * Ny * Nz; ++n){
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double value = 0.0;

        in >> x >> y >> z >> value;

        if(!in){
            throw runtime_error("Forcing file ended early");
        }

        const int i = n % Nx;
        const int j = (n / Nx) % Ny;
        const int k = n/ (Nx * Ny);

        global_f[global_index(i, j, k, Ny, Nz)] = value;
    }
}

/**
 * @brief Copy local solution data to the global solution array.
 * 
 * Each MPI process sends the values from its local subdomain to the position 
 * in the global solution array based on its global starting indices. This is
 * used on rank 0 when assembling the final global solution.
 */
void global2local(const vector<double>& global_f, vector<double>& local_f, 
                  const LocalDomain& dom, int Ny, int Nz){
    for (int i = 1; i <= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            for (int k = 1; k <= dom.nz; ++k){
                const int gi = dom.sx + (i - 1);
                const int gj = dom.sy + (j - 1);
                const int gk = dom.sz + (k - 1);

                local_f[local_index(i, j, k, dom.ny, dom.nz)] = 
                    global_f[global_index(gi, gj, gk, Ny, Nz)];
            }
        }
    }
}

/**
 *@brief Pack x-direction face of the local subdomain into a buffer
 *
 * Copies the values from the specified x-plane (iPlane) of the local
 * solution array into a contiguous buffer. The packed buffer is used
 * for MPI communication when exchanging spaced data with neighbouring
 * processes in the x-direction.
 *
 * @param u Local solution array
 * @param iPlane Local x-index of the face to pack
 * @param dom LocalDomain structure describing the local subdomain
 * @param buf Buffer storing the pakced face data
 */
void pack_face_x(const vector<double>& u, int iPlane, const LocalDomain& dom,
                 vector<double>& buf){
    buf.resize(dom.ny * dom.nz);
    int a = 0;
    for (int j = 1; j <= dom.ny; ++j){
        for (int k = 1; k <= dom.nz; ++k){
            buf[a++] = u[local_index(iPlane, j, k, dom.ny, dom.nz)];
        }
    }
}

/**
 * @brief Unpack received x-direction face data into the local spaced layer.
 *
 * Copies the values stored in the communication buffer into the specified
 * x-plane of the local solution array. This is used after MPI communication
 * to update the spcaed cells with data received from neighbouring processes
 * in the x-direction.
 *
 * @param u       Local solution array (including spaced cells)
 * @param iPlane  Local x-index of the spacecd layer to update
 * @param dom     LocalDomain structure describing the local subdomain
 * @param buf     Buffer containing received face data
 */
void unpack_face_x(vector<double>& u, int iPlane, const LocalDomain& dom, 
                   const vector<double>& buf){
    int a = 0;
    for (int j = 1; j <= dom.ny; ++j){
        for (int k = 1; k <= dom.nz; ++k){
            u[local_index(iPlane, j, k, dom.ny, dom.nz)] = buf[a++];
        }
    }
}


/**
 *@brief Pack y-direction face of the local subdomain into a buffer
 */
void pack_face_y(const vector<double>& u, int jPlane, const LocalDomain& dom,
                 vector<double>& buf){
    buf.resize(dom.nx * dom.nz);
    int a = 0;
    for (int i = 1; i <= dom.nx; ++i){
        for (int k = 1; k <= dom.nz; ++k){
            buf[a++] = u[local_index(i, jPlane, k, dom.ny, dom.nz)];
        }
    }
}

/**
 * @brief Unpack received y-direction face data into the local spaced layer.
 */
void unpack_face_y(vector<double>& u, int jPlane, const LocalDomain& dom, 
                   const vector<double>& buf){
    int a = 0;
    for (int i = 1; i <= dom.nx; ++i){
        for (int k = 1; k <= dom.nz; ++k){
            u[local_index(i, jPlane, k, dom.ny, dom.nz)] = buf[a++];
        }
    }
}

/**
 *@brief Pack z-direction face of the local subdomain into a buffer
 */
void pack_face_z(const vector<double>& u, int kPlane, const LocalDomain& dom,
                 vector<double>& buf){
    buf.resize(dom.nx * dom.ny);
    int a = 0;
    for (int i = 1; i <= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            buf[a++] = u[local_index(i, j, kPlane, dom.ny, dom.nz)];
        }
    }
}

/**
 * @brief Unpack received z-direction face data into the local spaced layer.
 */
void unpack_face_z(vector<double>& u, int kPlane, const LocalDomain& dom, 
                   const vector<double>& buf){
    int a = 0;
    for (int i = 1; i <= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            u[local_index(i, j, kPlane, dom.ny, dom.nz)] = buf[a++];
        }
    }
}

/**
 * @brief Exchange previously left spaced layers for Laplacian with neighbouring
 * MPI processes.
 *
 * Each MPI process sends the boundary values of its local subdomain to the six 
 * adjacent processes in the Cartesian plane and receives the corresponding data
 * into the spaced layers. The spaced cells are updates using MPI_Sendrecv after 
 * packing the boundary faces into buffers.
 *
 * @param u Local solution array including spaced cells
 * @param dom LocalDomain structure describing the local subdomain
 * @param xm Rank of neighbour in the negative x-direction
 * @param ym Rank of neighbour in the negative y-direction
 * @param zm Rank of neighbour in the negative z-direction
 * @param xp Rank of neighbour in the positive x-direction
 * @param yp Rank of neighbout in the positive y-direction
 * @param zp Rank of neighbour in the positive z-direction
 * @param cart_comm Cartesian MPI communicator used for communication.
 */
void swap_space(vector<double>& u, const LocalDomain& dom, int xm, int xp, 
                int ym, int yp, int zm, int zp, MPI_Comm cart_comm){
    vector<double> sendbuf;
    vector<double> recvbuf;
    MPI_Status status;

    // x-
    if (xm != MPI_PROC_NULL){
        pack_face_x(u, 1, dom, sendbuf);
        recvbuf = vector<double>(dom.ny * dom.nz, 0.0);
        MPI_Sendrecv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, xm, 100, recvbuf.data(),
                     recvbuf.size(), MPI_DOUBLE, xm, 101, cart_comm, &status);
        unpack_face_x(u, 0, dom, recvbuf);
    }

    // y- 
    if (ym != MPI_PROC_NULL){
        pack_face_y(u, 1, dom, sendbuf);
        recvbuf = vector<double>(dom.nx * dom.nz, 0.0);
        MPI_Sendrecv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, ym, 200, recvbuf.data(),
                     recvbuf.size(), MPI_DOUBLE, ym, 201, cart_comm, &status);
        unpack_face_y(u, 0, dom, recvbuf);
    }

    // z- 
    if (zm != MPI_PROC_NULL){
        pack_face_z(u, 1, dom, sendbuf);
        recvbuf = vector<double>(dom.nx * dom.ny, 0.0);
        MPI_Sendrecv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, zm, 300, recvbuf.data(),
                     recvbuf.size(), MPI_DOUBLE, zm, 301, cart_comm, &status);
        unpack_face_z(u, 0, dom, recvbuf);
    }
    
    // x+
    if (xp != MPI_PROC_NULL){
        pack_face_x(u, dom.nx, dom, sendbuf);
        recvbuf = vector<double>(dom.ny * dom.nz, 0.0);
        MPI_Sendrecv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, xp, 101, recvbuf.data(),
                     recvbuf.size(), MPI_DOUBLE, xp, 100, cart_comm, &status);
        unpack_face_x(u, dom.nx + 1, dom, recvbuf);
    }

    // y+
    if (yp != MPI_PROC_NULL){
        pack_face_y(u, dom.ny, dom, sendbuf);
        recvbuf = vector<double>(dom.nx * dom.nz, 0.0);
        MPI_Sendrecv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, yp, 201, recvbuf.data(),
                     recvbuf.size(), MPI_DOUBLE, yp, 200, cart_comm, &status);
        unpack_face_y(u, dom.ny + 1, dom, recvbuf);
    }

    // z+ 
    if (zp != MPI_PROC_NULL){
        pack_face_z(u, dom.nz, dom, sendbuf);
        recvbuf = vector<double>(dom.nx * dom.ny, 0.0);
        MPI_Sendrecv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, zp, 301, recvbuf.data(),
                     recvbuf.size(), MPI_DOUBLE, zp, 300, cart_comm, &status);
        unpack_face_z(u, dom.nz + 1, dom, recvbuf);
    }
}

/**
 * @brief Perform one Jacobi update on the locally owned grid points. 
 *
 * Update the points inside of the local subdomain using Jacobi. Global indices 
 * are reconstructed to detect global boundaries so Dirichlet boundary values remain 
 * unchanged. The previously kept spaced cells provide neighbourig values from the 
 * adjacent MPI processes.
 * 
 * @param u     Current local solution array
 * @param u_jac Temporary array storing the updated solution
 * @param f     Local forcing array
 * @param dom   LocalDomain structure describing the local subdomain
 * @param Nx, Ny, Nz    Global grid size in x, y, z direction
 * @param hx, hy, hz    Grid spacing in x, y, z direction
 */
void jacob_local(const vector<double>& u, vector<double>& u_jac,
                 const vector<double>& f, const LocalDomain& dom, int Nx, int Ny, 
                 int Nz, double hx, double hy, double hz){
    const double dx = 1.0 / (hx * hx);
    const double dy = 1.0 / (hy * hy);
    const double dz = 1.0 / (hz * hz);
    const double denom = 2.0 * (dx + dy + dz);

    for (int i = 1; i<= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            for (int k = 1; k <= dom.nz; ++k){
                const int gi = dom.sx + (i - 1);
                const int gj = dom.sy + (j - 1);
                const int gk = dom.sz + (k - 1);

                if (gi == 0 || gi == Nx - 1 || gj == 0 || gj == Ny - 1 ||
                     gk == 0 || gk == Nz - 1){
                    continue;
                 }

                 const int idx = local_index(i, j, k, dom.ny, dom.nz);
                 
                 u_jac[idx] =((u[local_index(i + 1, j, k, dom.ny, dom.nz)] + 
                               + u[local_index(i - 1, j, k, dom.ny, dom.nz)]) * dx + 
                              (u[local_index(i, j + 1, k, dom.ny, dom.nz)] + 
                               + u[local_index(i, j - 1, k, dom.ny, dom.nz)]) * dy + 
                              (u[local_index(i, j, k + 1, dom.ny, dom.nz)] + 
                               + u[local_index(i, j, k - 1, dom.ny, dom.nz)]) * dz
                               - f[idx]) / denom;
            }
        }
    }
}

/**
 * @brief Compute the global L2 residual norm of the Poisson equation.
 *
 * Each MPI rank computes the residual distribution over its local subdomain, and
 * then summed across all ranks using MPI_Allreduce to obtain the global L2 norm.
 * The boundary nodes are excluded since Dirichlet boundary conditions are fixed.
 *
 * @param u     Current local solution array
 * @param u_jac Temporary array storing the updated solution
 * @param f     Local forcing array
 * @param dom   LocalDomain structure describing the local subdomain
 * @param Nx, Ny, Nz    Global grid size in x, y, z direction
 * @param hx, hy, hz    Grid spacing in x, y, z direction
 * @param cart_comm     Cartesian communicator used for MPI reductions.
 *
 * @return Global L2 residual norm.
 */
double residual_global(const vector<double>& f, const vector<double>& u, const LocalDomain& dom, 
                       int Nx, int Ny, int Nz, double hx, double hy, double hz, MPI_Comm cart_comm){
    const double dx = 1.0 / (hx * hx);
    const double dy = 1.0 / (hy * hy);
    const double dz = 1.0 / (hz * hz);

    double local_sum = 0.0;

    for (int i = 1; i <= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            for (int k = 1; k <= dom.nz; ++k){
                const int gi = dom.sx + (i - 1);
                const int gj = dom.sy + (j - 1);
                const int gk = dom.sz + (k - 1);

                if (gi == 0 || gi == Nx - 1 || gj == 0 || gj == Ny - 1 ||
                    gk == 0 || gk == Nz - 1){
                    continue;
                }

                const int idx = local_index(i, j, k, dom.ny, dom.nz);

                const double lapx = (u[local_index(i + 1, j, k, dom.ny, dom.nz)] - 2.0 * u[idx] + 
                                     u[local_index(i - 1, j, k, dom.ny, dom.nz)]) * dx;
                const double lapy = (u[local_index(i, j + 1, k, dom.ny, dom.nz)] - 2.0 * u[idx] + 
                                     u[local_index(i, j - 1, k, dom.ny, dom.nz)]) * dy;
                const double lapz = (u[local_index(i, j, k + 1, dom.ny, dom.nz)] - 2.0 * u[idx] + 
                                     u[local_index(i, j, k - 1, dom.ny, dom.nz)]) * dz;
                const double r = f[idx] - (lapx + lapy + lapz);
                local_sum += r * r;
            }
        }
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

    return sqrt(global_sum);
}

/**
 * @brief Pack the local solution values to buffer.
 *
 * This function copies the interior portion of the local solution array into a 1D buffer.
 * The packed buffer can then be sent to another process for assembling the global solution.
 * 
 * @param u     Local solution vector.
 * @param dom   Local domain information.
 * @param buf   Output buffer storing the packed local block
 * @param buf   Output buffer storing the packed local block
 */
void pack_block(const vector<double>& u, const LocalDomain& dom, vector<double>& buf){
    buf.resize(dom.nx * dom.ny * dom.nz);
    int a = 0;
    for (int i = 1; i <= dom.nx; ++i){
        for (int j = 1; j <= dom.ny; ++j){
            for (int k = 1; k <= dom.nz; ++k){
                buf[a++] = u[local_index(i, j, k, dom.ny, dom.nz)];
            }
        }
    }
}


/** 
 * @brief Unpack a received local block into the global solution array
 *
 * This function copies values from a packed buffer into the appropriate positions
 * in the global solution vector using the global indices of the local domain. It
 * is typically used on the root process after receiving local solution blocks from
 * other MPI ranks.
 *
 * @param buf Buffer containing the packed local solution block.
 * @param global_u Gloabl solution vector storing the full domain.
 * @param dom  Local domain information
 * @param Ny Global number of grid points in y direction
 * @param Nz Gloabl number of grid points in z direction
 */
void unpack_block(const vector<double>& buf, vector<double>& global_u, const LocalDomain& dom,
                  int Ny, int Nz){
    int a = 0;
    for (int i = 0; i <dom.nx; ++i){
        for (int j = 0; j < dom.ny; ++j){
            for (int k = 0; k < dom.nz; ++k){
                const int gi = dom.sx + i;
                const int gj = dom.sy + j;
                const int gk = dom.sz + k;
                global_u[global_index(gi, gj, gk, Ny, Nz)] = buf[a++];
            }
        }
    }
}

/**
 * @brief Write solution_mpi.txt in the required single-file format.
 */
void write_solution(const string& filename, const vector<double>& global_u, 
                    int Nx, int Ny, int Nz, double hx, double hy, double hz){
    ofstream file(filename);
    if(!file){
        throw runtime_error("Failed to open output file: " + filename);
    }

    file << Nx << " " << Ny << " " << Nz << "\n";
    file << fixed << setprecision(8);
    
    for (int k = 0; k < Nz; ++k){
        for (int j = 0; j < Ny; ++j){
            for (int i = 0; i < Nx; ++i){
                const double x = i * hx;
                const double y = j * hy;
                const double z = k * hz;
                const double val = global_u[global_index(i, j, k, Ny, Nz)];

                file << setw(14) << x
                     << setw(14) << y
                     << setw(14) << z
                     << setw(18) << val << "\n";
            }
        }
    }
}

/**
 * @brief Broadcase three integers for later use of assigning Nx, Ny, Nz
 */
void bcast_int(int& a, int& b, int& c, int root, MPI_Comm comm){
    int vals[3] = {a, b, c};
    MPI_Bcast(vals, 3, MPI_INT, root, comm);
    a = vals[0];
    b = vals[1];
    c = vals[2];
}

/**
 * @brief Broadcast three doubles.
 */
 void bcast_double(double& a, double& b, double& c, int root, MPI_Comm comm){
    double vals[3] = {a, b, c};
    MPI_Bcast(vals, 3, MPI_DOUBLE, root, comm);
    a = vals[0];
    b = vals[1];
    c = vals[2];
 }

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        Options opt = respond(argc, argv);

        if (opt.help){
            if (world_rank == 0){
                help();
            }
            MPI_Finalize();
            return 0;
        }

        if (opt.Px * opt.Py * opt.Pz != world_size){
            throw runtime_error("Px * Py * Pz must equal the number of MPI ranks.");
        }

        // Create Cartesian communicator
        int dims[3] = {opt.Px, opt.Py, opt.Pz};
        int periods[3] = {0, 0, 0};
        MPI_Comm cart_comm;
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

        int rank = 0;
        MPI_Comm_rank(cart_comm, &rank);

        int coords[3] = {0, 0, 0};
        MPI_Cart_coords(cart_comm, rank, 3, coords);

        int xm = MPI_PROC_NULL, xp = MPI_PROC_NULL;
        int ym = MPI_PROC_NULL, yp = MPI_PROC_NULL;
        int zm = MPI_PROC_NULL, zp = MPI_PROC_NULL;
        MPI_Cart_shift(cart_comm, 0, 1, &xm, &xp);
        MPI_Cart_shift(cart_comm, 1, 1, &ym, &yp);
        MPI_Cart_shift(cart_comm, 2, 1, &zm, &zp);

        int Nx = opt.Nx;
        int Ny = opt.Ny;
        int Nz = opt.Nz;

        //Test mode: fill defaults if not set
        if (opt.if_test){
            int dNx = 0;
            int dNy = 0;
            int dNz = 0;

            set_default(opt.test, dNx, dNy, dNz);

            if(!opt.Nx_set){
                Nx = dNx;
            }
            if(!opt.Ny_set){
                Ny = dNy;
            }
            if(!opt.Nz_set){
                Nz = dNz;
            }

            if (Nx < 2 || Ny < 2 || Nz < 2){
                throw runtime_error("Nx, Ny and Nz must each be at least 2.");
            }
        }

        double hx = 0.0;
        double hy = 0.0;
        double hz = 0.0;

        vector<double> global_f;

        if(opt.if_test){
            hx = 1.0 / (Nx - 1);
            hy = 1.0 / (Ny - 1);
            hz = 1.0 / (Nz - 1);
        } else {
            if (rank == 0){
                read_forcing_file(opt.forcing_file, global_f, Nx, Ny, Nz, hx, hy, hz);
            }

            bcast_int(Nx, Ny, Nz, 0, cart_comm);
            bcast_double(hx, hy, hz, 0, cart_comm);
        }

        if(opt.Px > Nx || opt.Py > Ny || opt.Pz > Nz){
            throw runtime_error("Process decomposition is too fine for the chosen grid.");
        }

        // Build local subdomain
        LocalDomain dom = build_local_domain(Nx, Ny, Nz, opt.Px, opt.Py, opt.Pz,
                                             coords[0], coords[1], coords[2]);
        vector<double> u((dom.nx + 2) * (dom.ny + 2) * (dom.nz + 2), 0.0);
        vector<double> u_jac((dom.nx + 2) * (dom.ny + 2) * (dom.nz + 2), 0.0);
        vector<double> f((dom.nx + 2) * (dom.ny + 2) * (dom.nz + 2), 0.0);

        if(opt.if_test){
            test_case_local(opt.test, u, u_jac, f, dom, Nx, Ny, Nz, hx, hy, hz);
        } else {
            if(rank == 0){
                // Root copies its own sub-block
                global2local(global_f, f, dom, Ny, Nz);

                // Send sub-blocks to other ranks
                for (int r = 1; r < world_size; ++r){
                    int rc[3] = {0, 0, 0};
                    MPI_Cart_coords(cart_comm, r, 3, rc);

                    LocalDomain rd = build_local_domain(Nx, Ny, Nz, opt.Px, opt.Py, opt.Pz, rc[0], rc[1], rc[2]);

                    vector<double> sendbuf(rd.nx * rd.ny * rd.nz, 0.0);
                    int a = 0;
                    for (int i = 0; i < rd.nx; ++i){
                        for (int j = 0; j < rd.ny; ++j){
                            for (int k = 0; k < rd.nz; ++k){
                                int gi = rd.sx + i;
                                int gj = rd.sy + j;
                                int gk = rd.sz + k;
                                sendbuf[a++] = global_f[global_index(gi, gj, gk, Ny, Nz)];
                            }
                        }
                    }

                    MPI_Send(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, r, 500, cart_comm);
                }
            } else {
                vector<double> recvbuf(dom.nx * dom.ny * dom.nz, 0.0);

                MPI_Recv(recvbuf.data(), recvbuf.size(), MPI_DOUBLE, 0, 500, cart_comm, MPI_STATUS_IGNORE);

                int a = 0;
                for (int i = 1; i <= dom.nx; ++i){
                    for (int j = 1; j <= dom.ny; ++j){
                        for (int k = 1; k <= dom.nz; ++k){
                            f[local_index(i, j, k, dom.ny, dom.nz)] = recvbuf[a++];
                        }
                    }
                }

                for (int i = 1; i <= dom.nx; ++i){
                    for (int j = 1; j <= dom.ny; ++j){
                        for (int k = 1; k <= dom.nz; ++k){
                            int idx = local_index(i, j, k, dom.ny, dom.nz);
                            u[idx] = 0.0;
                            u_jac[idx] = 0.0;
                        }
                    }
                }
            }
        }

        swap_space(u, dom, xm, xp, ym, yp, zm, zp, cart_comm);


        double residual = residual_global(f, u, dom, Nx, Ny, Nz, hx, hy, hz, cart_comm);
        const int max_iter = 1000000;
        int iter = 0;

        auto start = chrono::high_resolution_clock::now();

        double swap_time = 0.0;
        double jacloc_time = 0.0;
        double residual_time = 0.0;

        while (residual > opt.epsilon && iter < max_iter){
            auto t_swap_s = chrono::high_resolution_clock::now();
            swap_space(u, dom, xm, xp, ym, yp, zm, zp, cart_comm);
            auto t_swap_e = chrono::high_resolution_clock::now();
            swap_time += chrono::duration<double>(t_swap_e - t_swap_s).count();

            auto t_jacloc_s = chrono::high_resolution_clock::now();
            jacob_local(u, u_jac, f, dom, Nx, Ny, Nz, hx, hy, hz);
            auto t_jacloc_e = chrono::high_resolution_clock::now();
            jacloc_time += chrono::duration<double>(t_jacloc_e - t_jacloc_s).count();
            u.swap(u_jac);

            auto t_residual_s = chrono::high_resolution_clock::now();
            residual = residual_global(f, u, dom, Nx, Ny, Nz, hx, hy, hz, cart_comm);
            auto t_residual_e = chrono::high_resolution_clock::now();
            residual_time += chrono::duration<double>(t_residual_e - t_residual_s).count();
            ++iter;
        }

        auto end = chrono::high_resolution_clock::now();
        double time = chrono::duration<double>(end - start).count();

        if (iter == max_iter && residual > opt.epsilon){
            throw runtime_error("Jacobi iteration did not converge within max iteration.");
        }

        vector<double> sendbuf;
        pack_block(u, dom, sendbuf);

        if (rank == 0){
            vector<double> global_u(Nx * Ny * Nz, 0.0);

            unpack_block(sendbuf, global_u, dom, Ny, Nz);

            for (int r = 1; r < world_size; ++r){
                int rc[3] = {0, 0, 0};
                MPI_Cart_coords(cart_comm, r, 3, rc);

                LocalDomain rd = build_local_domain(Nx, Ny, Nz, opt.Px, opt.Py, opt.Pz,
                                                    rc[0], rc[1], rc[2]);

                vector<double> recvbuf(rd.nx * rd.ny * rd.nz, 0.0);

                MPI_Recv(recvbuf.data(), recvbuf.size(), MPI_DOUBLE, r, 600, cart_comm,
                         MPI_STATUS_IGNORE);

                unpack_block(recvbuf, global_u, rd, Ny, Nz);
            }

            cout << "Jacobi compute time: " << jacloc_time << endl;
            cout << "Halo communication time: " << swap_time << endl;
            cout << "Residual evaluation time: " << residual_time << endl;

            write_solution("solution.txt", global_u, Nx, Ny, Nz, hx, hy, hz);

            cout << scientific << setprecision(10);
            cout << "Convergence time: " << time << "\n";
            cout << "Iterations: " << iter << "\n";
            cout << "Residual: " << residual << "\n";
        } else {
            MPI_Send(sendbuf.data(), sendbuf.size(), MPI_DOUBLE, 0, 600, cart_comm);
        }

        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return 0;
        } catch (const exception& e){
            if (world_rank == 0){
                cerr << "Error: " << e.what() << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
}
