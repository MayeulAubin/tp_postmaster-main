#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream> 
#include <random>
#include <sstream>
#include <vector>
#include <Kokkos_Core.hpp>

#include "convection.hpp"

namespace conv_variables {
    constexpr int nvar = 6;  // Number of variables (e.g., density, momentum, energy, etc.)
    enum : int {  // Indices for different variables
        ID = 0,   // Variable identifier for density
        IU = 1,   // Variable identifier for x-momentum
        IV = 2,   // Variable identifier for y-momentum
        IW = 3,   // Variable identifier for z-momentum
        IE = 4,   // Variable identifier for total energy
        IG = 5    // Variable identifier for gravitational potential energy
    };
    int nx, ny, nz;  // Grid dimensions and simulation parameters
    double Lx, Ly, Lz, gam, cv, grav;  // Physical constants and parameters
    double tau, T_bottom, rho_bottom, dTdz, dx, dy, dz;  // Additional parameters for temperature and grid spacing
}

// This function fills an output data array with the state variables of the fluid simulation
// at each grid cell. It computes the density, velocity components, kinetic energy, gravitational energy,
// and temperature for each cell in the grid and stores these values in the provided data array.
void fill_data(Kokkos::View<double**, Kokkos::HostSpace> data, Kokkos::View<double****, Kokkos::LayoutLeft, Kokkos::HostSpace> U, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> xc, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> yc, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> zc) {
    using namespace conv_variables;

    

    // Loop over all grid cells 
    Kokkos::parallel_for("fill_data", 
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>({1, 1, 1}, {nx+1, ny+1, nz+1}), 
        KOKKOS_LAMBDA (int i, int j, int k){
            // Loop variables
            int index;
            // Temporary variables for calculations
            double rhoc, uc, vc, wc, ekinc, egc, Tc;
            // Retrieve density and velocity components
            rhoc = U(i, j, k, ID);              // Density
            uc = U(i, j, k, IU) / rhoc;         // X-velocity
            vc = U(i, j, k, IV) / rhoc;         // Y-velocity
            wc = U(i, j, k, IW) / rhoc;         // Z-velocity

            // Calculate kinetic and gravitational energy
            ekinc = 0.5 * (uc * uc + vc * vc + wc * wc) * rhoc;  // Kinetic energy
            egc = rhoc * U(i, j, k, IG);          // Gravitational energy

            // Calculate temperature
            Tc = (U(i, j, k, IE) - ekinc - egc) / (cv * rhoc);

            // Fill the data array with computed values
            index = nx * ny * (k - 1) + nx * (j - 1) + i -1;
            data(index,0) = xc(i);  // X-coordinate
            data(index,1) = yc(j);  // Y-coordinate
            data(index,2) = zc(k);  // Z-coordinate
            data(index,3) = U(i, j, k, ID);  // Density
            data(index,4) = U(i, j, k, IU);  // X-momentum
            data(index,5) = U(i, j, k, IV);  // Y-momentum
            data(index,6) = U(i, j, k, IW);  // Z-momentum
            data(index,7) = U(i, j, k, IE);  // Internal energy
            data(index,8) = Tc;              // Temperature
        });
        
}

// This function writes the output of the fluid simulation to a CSV file at specified time steps.
// It calculates the maximum kinetic energy across the grid, prints the current time step and kinetic energy,
// and saves the simulation state to a file if the output frequency condition is met.
void write_output(int it, int freq_output, Kokkos::View<double****, Kokkos::LayoutLeft, Kokkos::HostSpace>  U, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> xc, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> yc, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> zc) {
    using namespace conv_variables;

    double ekin_max;
    int iout; // Output index
    std::string output_file; // Output file name

    // Calculate the maximum kinetic energy across the grid
    Kokkos::parallel_reduce("find_max_output", 
        Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>({1, 1, 1}, {nx+1, ny+1, nz+1}), 
        KOKKOS_LAMBDA (int i, int j, int k, double& Ekin){
                Ekin = std::max(Ekin,0.5 * (U(i, j, k, IU)*U(i, j, k, IU) + U(i, j, k, IV)*U(i, j, k, IV) + U(i, j, k, IW)*U(i, j, k, IW))/U(i, j, k, ID));
        },
        Kokkos::Max<double>(ekin_max));
     

    // Print the current time step and kinetic energy to the console
    std::cout << "timestep: " << it << " kinetic energy: " << ekin_max << '\n';

    // Check if it's time to write output based on the frequency
    if (it % freq_output == 0) {
        iout = it / freq_output; // Calculate output index
        std::cout << "write csv output: " << iout << '\n'; // Inform about the output being generated

        Kokkos::View<double**, Kokkos::HostSpace> data("data_output", nx * ny * nz, nvar + 3);

        // Fill the data array with the current simulation state
        fill_data(data, U, xc, yc, zc);

        // Generate the output file name
        std::stringstream ss_iout;
        ss_iout << std::setfill('0') << std::setw(6) << iout;
        output_file = "output.csv." + ss_iout.str();

        // Open the output file for writing
        std::ofstream outfile(output_file);

        // Write the header for the CSV file
        outfile << "x,y,z,rho,rhou,rhov,rhow,rhoE,temperature\n";

        // Write the data to the output file
        for (int i = 0; i < nx * ny * nz; ++i) {
            outfile << std::fixed << std::setprecision(5) << data(i, 0);
            for (int j = 1; j < 9; ++j) {
                outfile << ',' << data(i, j);
            }
            outfile << '\n';
        }

        // Close the output file
        outfile.close();
    }
}

// This function initializes the conditions for a fluid simulation in a 2D grid.
// It sets the initial values for density, momentum, total energy, and gravitational potential energy
// at each grid cell based on specified boundary conditions and temperature gradients.
void compute_initial_condition(Kokkos::View<double****> U, const Kokkos::View<double*> xc, const Kokkos::View<double*> yc, const Kokkos::View<double*> zc) {
    using namespace conv_variables;


    Kokkos::parallel_for("init_bottom", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({1, 1}, {nx+1, ny+1}), 
        KOKKOS_LAMBDA (int i, int j){
            U(i, j, 1, ID) = rho_bottom;  // Set density at the bottom of the domain
            U(i, j, 1, IU) = 0.0;          // Set initial X-momentum at the bottom to zero
            U(i, j, 1, IV) = 0.0;          // Set initial Y-momentum at the bottom to zero
            U(i, j, 1, IW) = 0.0;          // Set initial Z-momentum at the bottom to zero
            U(i, j, 1, IE) = rho_bottom * cv * T_bottom + rho_bottom * (-grav * zc[1]);  // Total energy at the bottom, considering gravitational potential
            U(i, j, 1, IG) = -grav * zc[1];  // Gravitational potential energy at the bottom
        });

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // Recursive initialization of the rest of the domain
    // Kokkos::parallel_for("init_domain", 
    //     Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({1, 1}, {nx+1, ny+1}), 
    //     KOKKOS_LAMBDA (int i, int j){
    for(int i = 1; i <= nx; ++i)
        for(int j = 1; j <= ny; ++j)
            for (int k = 2; k <= nz; ++k) {
                // Temporary variables for temperature and density calculations
                double Tl, Tr, rhol, rhor, ur, vr, wr;
                // Calculate left and right temperatures based on the vertical position
                Tl = T_bottom + dTdz * (zc[k - 1] - zc[1]);  // Temperature at the previous grid cell (k-1)
                Tr = T_bottom + dTdz * (zc[k] - zc[1]);      // Temperature at the current grid cell (k)

                rhol = U(i, j, k - 1, ID);  // Retrieve density from the previous grid cell

                // Calculate the density at the current grid cell using the temperature gradient
                rhor = rhol * ((gam - 1.0) * cv * Tl + 0.5 * grav * dz) /
                             ((gam - 1.0) * cv * Tr - 0.5 * grav * dz);

                ur = 0.0;  // Set initial x-velocity to zero
                vr = 0.0;  // Set initial y-velocity to zero

                // Generate a small random value for z-velocity
                wr = 1E-3 * dis(gen);  // Scale the random value

                // Fill the state variables for the current grid cell
                U(i, j, k, ID) = rhor;  // Density
                U(i, j, k, IU) = rhor * ur;  // X-momentum
                U(i, j, k, IV) = rhor * vr;  // Y-momentum
                U(i, j, k, IW) = rhor * wr;  // Z-momentum
                U(i, j, k, IE) = rhor * cv * Tr + rhor * (-grav * zc[k]) + 0.5 * (ur * ur + vr * vr + wr * wr) * rhor;  // Total energy
                U(i, j, k, IG) = -grav * zc[k];  // Gravitational potential energy
            }
        // });
    
}

// This function computes boundary conditions for a fluid simulation in a 2D grid.
// It extrapolates values for the bottom and top boundaries based on the values from the
// two adjacent layers, and sets periodic boundary conditions in the x/y-directions.
// The calculations involve density, velocity, kinetic energy, internal energy, and
// temperature, taking into account gravitational effects.
void compute_boundary_condition(Kokkos::View<double****>  U, const Kokkos::View<double*> xc, const Kokkos::View<double*> yc, const Kokkos::View<double*> zc) {
    using namespace conv_variables;


    // Extrapolation for the bottom boundary (k = 0)
    Kokkos::parallel_for("BC_bottom", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({1, 1}, {nx+1, ny+1}), 
        KOKKOS_LAMBDA (int i, int j){
            // Temporary variables for calculations
            double rho2, u2, v2, w2, ekin2, eg2, T2;
            double rho1, u1, v1, w1, ekin1, eg1, T1;
            double T0, rho0, ekin0, eg0;
            // Get the values from the second layer
            rho2 = U(i, j, 2, ID);
            u2 = U(i, j, 2, IU) / rho2;
            v2 = U(i, j, 2, IV) / rho2;
            w2 = U(i, j, 2, IW) / rho2;
            ekin2 = 0.5 * (u2 * u2 + v2 * v2 + w2 * w2) * rho2;
            eg2 = rho2 * U(i, j, 2, IG);
            T2 = (U(i, j, 2, IE) - ekin2 - eg2) / (cv * rho2);

            // Get the values from the first layer
            rho1 = U(i, j, 1, ID);
            u1 = U(i, j, 1, IU) / rho1;
            v1 = U(i, j, 1, IV) / rho1;
            w1 = U(i, j, 1, IW) / rho1;
            ekin1 = 0.5 * (u1 * u1 + v1 * v1 + w1 * w1) * rho1;
            eg1 = rho1 * U(i, j, 1, IG);
            T1 = (U(i, j, 1, IE) - ekin1 - eg1) / (cv * rho1);

            // Calculate the extrapolated values
            T0 = 2.0 * T1 - T2;
            rho0 = rho1 * ((gam - 1.0) * cv * T1 - 0.5 * grav * dz) /
                        ((gam - 1.0) * cv * T0 + 0.5 * grav * dz);

            // Set boundary conditions for the bottom layer
            U(i, j, 0, ID) = rho0;
            U(i, j, 0, IU) = rho0 * u1;
            U(i, j, 0, IV) = rho0 * v1;
            U(i, j, 0, IW) = -rho0 * w1;
            ekin0 = ekin1 / rho1 * rho0;
            eg0 = rho0 * (-grav * zc[1]);
            U(i, j, 0, IE) = rho0 * cv * T0 + ekin0 + eg0;
            U(i, j, 0, IG) = -grav * zc[1];
        });

    // Extrapolation for the top boundary (k = nz + 1)
    Kokkos::parallel_for("BC_top", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({1, 1}, {nx+1, ny+1}), 
        KOKKOS_LAMBDA (int i, int j){
            // Temporary variables for calculations
            double rho2, u2, v2, w2, ekin2, eg2, T2;
            double rho1, u1, v1, w1, ekin1, eg1, T1;
            double T0, rho0, ekin0, eg0;
            rho2 = U(i, j, nz - 1, ID);
            u2 = U(i, j, nz - 1, IU) / rho2;
            v2 = U(i, j, nz - 1, IV) / rho2;
            w2 = U(i, j, nz - 1, IW) / rho2;
            ekin2 = 0.5 * (u2 * u2 + v2 * v2 + w2 * w2) * rho2;
            eg2 = rho2 * U(i, j, nz - 1, IG);
            T2 = (U(i, j, nz - 1, IE) - ekin2 - eg2) / (cv * rho2);

            rho1 = U(i, j, nz, ID);
            u1 = U(i, j, nz, IU) / rho1;
            v1 = U(i, j, nz, IV) / rho1;
            w1 = U(i, j, nz, IW) / rho1;
            ekin1 = 0.5 * (u1 * u1 + v1 * v1 + w1 * w1) * rho1;
            eg1 = rho1 * U(i, j, nz, IG);
            T1 = (U(i, j, nz, IE) - ekin1 - eg1) / (cv * rho1);

            T0 = 2.0 * T1 - T2;
            rho0 = rho1 * ((gam - 1.0) * cv * T1 + 0.5 * grav * dz) /
                        ((gam - 1.0) * cv * T0 - 0.5 * grav * dz);

            // Set boundary conditions for the top layer
            U(i, j, nz + 1, ID) = rho0;
            U(i, j, nz + 1, IU) = rho0 * u1;
            U(i, j, nz + 1, IV) = rho0 * v1;
            U(i, j, nz + 1, IW) = -rho0 * w1;
            ekin0 = ekin1 / rho1 * rho0;
            eg0 = rho0 * (-grav * zc[nz + 1]);
            U(i, j, nz + 1, IE) = rho0 * cv * T0 + ekin0 + eg0;
            U(i, j, nz + 1, IG) = -grav * zc[nz + 1];
        });

    // Periodic boundary conditions in the x-direction
    Kokkos::parallel_for("BC_x", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>({1, 0, 0}, {ny+1, nz+2, nvar}), 
        KOKKOS_LAMBDA (int j, int k, int ivar){
            U(0, j, k, ivar) = U(nx, j, k, ivar);      // Left boundary
            U(nx + 1, j, k, ivar) = U(1, j, k, ivar);  // Right boundary
        });

    // Periodic boundary conditions in the y-direction
    Kokkos::parallel_for("BC_y", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>({0, 0, 0}, {nx+2, nz+2, nvar}), 
        KOKKOS_LAMBDA (int i, int k, int ivar){
            U(i, 0, k, ivar) = U(i, ny, k, ivar);      // Left boundary
            U(i, ny + 1, k, ivar) = U(i, 1, k, ivar);  // Right boundary
        });
}

// This function computes the local time step for a numerical simulation
// based on the CFL condition. It iterates over a grid defined by nx and ny,
// calculating the density, velocities, kinetic and gravitational energy,
// pressure, and speed of sound at each grid cell. The global time step
// is updated to the minimum value found across all grid cells to ensure
// stability in the simulation.
double compute_timestep(Kokkos::View<double****> U) {
    using namespace conv_variables;

    double dt;
    
    // Loop over the grid cells to compute the local time step
    Kokkos::parallel_reduce("compute_dt", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>({1, 1, 1}, {nx+1, ny+1, nz+1}), 
        KOKKOS_LAMBDA (int i, int j, int k, double& dt_loc){
            // Local variables for time step calculation
            double rhoc, uc, vc, wc, ekinc, egc, pc, ac;
            // Calculate the density and velocity components
            rhoc = U(i, j, k, ID);
            uc = U(i, j, k, IU) / rhoc;  // X-velocity
            vc = U(i, j, k, IV) / rhoc;  // Y-velocity
            wc = U(i, j, k, IW) / rhoc;  // Z-velocity

            // Calculate kinetic and gravitational energy
            ekinc = 0.5 * (uc * uc + vc * vc + wc * wc) * rhoc;
            egc = rhoc * U(i, j, k, IG);

            // Calculate pressure and sound speed
            pc = (U(i, j, k, IE) - ekinc - egc) * (gam - 1.0);
            ac = Kokkos::sqrt(gam * pc / rhoc);  // Speed of sound

            // Calculate the local time step based on CFL condition
            dt_loc = Kokkos::min(dt_loc,Kokkos::min(Kokkos::min(dx, dy), dz) / (ac + Kokkos::sqrt(uc * uc + vc * vc + wc * wc)));
        },
        Kokkos::Min<double>(dt));
        return dt;
}

// This function computes the flux at a face between two states in a convection problem.
// It takes in left and right state vectors, gravitational acceleration, and outputs the flux vector.
// The calculations involve determining various properties (density, velocity, energy) for both states,
// as well as face-centered properties, and then applying a numerical flux method based on wave speeds.
template<std::size_t nvar>
void compute_flux(const Kokkos::Array<double, nvar> &Ucl, const Kokkos::Array<double, nvar> &Ucr, double gravdx, Kokkos::Array<double, nvar> &flux) {
    using namespace conv_variables;

    // Local variables for left state calculations
    double rhol, ul, vl, wl, ekinl, egl, pl, al;
    // Local variables for right state calculations
    double rhor, ur, vr, wr, ekinr, egr, pr, ar;
    // Additional variables for flux calculations
    double aface, ustar, theta, pstar;

    // Calculate left state properties
    rhol = Ucl[ID];
    ul = Ucl[IU] / rhol;
    vl = Ucl[IV] / rhol;
    wl = Ucl[IW] / rhol;
    ekinl = 0.5 * (ul * ul + vl * vl + wl * wl) * rhol;
    egl = rhol * Ucl[IG];
    pl = (Ucl[IE] - ekinl - egl) * (gam - 1.0);
    al = rhol * Kokkos::sqrt(gam * pl / rhol);

    // Calculate right state properties
    rhor = Ucr[ID];
    ur = Ucr[IU] / rhor;
    vr = Ucr[IV] / rhor;
    wr = Ucr[IW] / rhor;
    ekinr = 0.5 * (ur * ur + vr * vr + wr * wr) * rhor;
    egr = rhor * Ucr[IG];
    pr = (Ucr[IE] - ekinr - egr) * (gam - 1.0);
    ar = rhor * Kokkos::sqrt(gam * pr / rhor);

    // Calculate face-centered properties
    aface = 1.1 * Kokkos::max(al, ar);  // Face speed
    ustar = 0.5 * (ul + ur) - 0.5 * (pr - pl - 0.5 * (rhol + rhor) * gravdx) / aface;
    theta = Kokkos::min(Kokkos::fabs(ustar) / Kokkos::max(al / rhol, ar / rhor), 1.0);  // Non-dimensional speed
    pstar = 0.5 * (pl + pr) - 0.5 * (ur - ul) * aface * theta;  // Star pressure

    // Calculate fluxes based on the wave speed (ustar]
    if (ustar > 0.0) {
        flux[ID] = ustar * Ucl[ID];
        flux[IU] = ustar * Ucl[IU] + pstar;
        flux[IV] = ustar * Ucl[IV];
        flux[IW] = ustar * Ucl[IW];
        flux[IE] = ustar * Ucl[IE] + pstar * ustar;
    } else {
        flux[ID] = ustar * Ucr[ID];
        flux[IU] = ustar * Ucr[IU] + pstar;
        flux[IV] = ustar * Ucr[IV];
        flux[IW] = ustar * Ucr[IW];
        flux[IE] = ustar * Ucr[IE] + pstar * ustar;
    }
}

// This function swaps the x and y or z velocity components in a given array.
// It takes a one-dimensional array 'arr' as input/output and exchanges and the y/z direction
// the values at indices IU and IV/IW, which represent the x and y/z velocities respectively.
template<std::size_t nvar>
void swap_direction(Kokkos::Array<double, nvar> &arr, int IVW) {
    double temp;  // Temporary variable for swapping
    // Swap the x and y velocity components
    temp = arr[conv_variables::IU];         // Store the x-velocity in a temporary variable
    arr[conv_variables::IU] = arr[IVW];     // Assign the y/z- velocity to the x-velocity position
    arr[IVW] = temp;       // Assign the stored x-velocity to the y/z-velocity position
}

// This function computes the numerical fluxes for a two-dimensional grid
// using a finite volume method. It updates the state variables Unew based on the
// previous state Uold, considering both x and y directional fluxes, as well as
// gravitational and thermal source terms. The computations involve updating
// the energy and temperature based on fluid dynamics principles and the
// conservation equations.
void compute_kernel(Kokkos::View<double****> Uold, Kokkos::View<double****> Unew, double dt, const Kokkos::View<double*> xc, const Kokkos::View<double*> yc, const Kokkos::View<double*> zc) {
    using namespace conv_variables;


    // Loop over the grid cells
    Kokkos::parallel_for("compute_kernel", 
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>({1, 1, 1}, {nx+1, ny+1, nz+1}), 
        KOKKOS_LAMBDA (int i, int j, int k){
            int ivar;
            // Variables for energy and temperature calculations
            double egc, ekinc, rhoc, Tc, Teq, Tnew, uc, vc, wc;
            // State vectors for left and right states and computed flux
            Kokkos::Array<double, nvar> Ul, Ur, flux;
            // Compute fluxes in the x direction (left and right)
            // Left flux in x direction
            for (ivar = 0; ivar < nvar; ++ivar){
                Ul[ivar] = Uold(i-1, j, k, ivar);
                Ur[ivar] = Uold(i, j, k, ivar);
            }
            // Init flux
            for (ivar = 0; ivar < nvar; ++ivar){
                flux[ivar] = 0.;
            }
            compute_flux(Ul, Ur, 0.0, flux);
            for (ivar = 0; ivar < nvar; ++ivar) {
                Unew(i, j, k, ivar) += dt / dx * flux[ivar];
            }
            // Right flux in x direction
            for (ivar = 0; ivar < nvar; ++ivar){
                Ul[ivar] = Uold(i, j, k, ivar);
                Ur[ivar] = Uold(i+1, j, k, ivar);
            }
            compute_flux(Ul, Ur, 0.0, flux);
            for (ivar = 0; ivar < nvar; ++ivar) {
                Unew(i, j, k, ivar) -= dt / dx * flux[ivar];
            }
            // Compute fluxes in the y direction (up and down)
            // Left flux in y direction
            for (ivar = 0; ivar < nvar; ++ivar){
                Ul[ivar] = Uold(i, j-1, k, ivar);
                Ur[ivar] = Uold(i, j, k, ivar);
            }
            swap_direction(Ul, IV);
            swap_direction(Ur, IV);
            compute_flux(Ul, Ur, 0.0, flux);
            swap_direction(flux, IV);
            for (ivar = 0; ivar < nvar; ++ivar) {
                Unew(i, j, k, ivar) += dt / dy * flux[ivar];
            }
            // Right flux in y direction
            for (ivar = 0; ivar < nvar; ++ivar){
                Ul[ivar] = Uold(i, j, k, ivar);
                Ur[ivar] = Uold(i, j+1, k, ivar);
            }
            swap_direction(Ul, IV);
            swap_direction(Ur, IV);
            compute_flux(Ul, Ur, 0.0, flux);
            swap_direction(flux, IV);
            for (ivar = 0; ivar < nvar; ++ivar) {
                Unew(i, j, k, ivar) -= dt / dy * flux[ivar];
            }
            // Compute fluxes in the z direction (up and down)
            // Left flux in z direction
            for (ivar = 0; ivar < nvar; ++ivar){
                Ul[ivar] = Uold(i, j, k-1, ivar);
                Ur[ivar] = Uold(i, j, k, ivar);
            }
            swap_direction(Ul, IW);
            swap_direction(Ur, IW);
            compute_flux(Ul, Ur, grav * dz, flux);
            swap_direction(flux, IW);
            for (ivar = 0; ivar < nvar; ++ivar) {
                Unew(i, j, k, ivar) += dt / dz * flux[ivar];
            }
            // Right flux in z direction
            for (ivar = 0; ivar < nvar; ++ivar){
                Ul[ivar] = Uold(i, j, k, ivar);
                Ur[ivar] = Uold(i, j, k+1, ivar);
            }
            swap_direction(Ul, IW);
            swap_direction(Ur, IW);
            compute_flux(Ul, Ur, grav * dz, flux);
            swap_direction(flux, IW);
            for (ivar = 0; ivar < nvar; ++ivar) {
                Unew(i, j, k, ivar) -= dt / dz * flux[ivar];
            }
            // Gravity source term
            Unew(i, j, k, IW) += dt * 0.25 * (Uold(i, j, k-1, ID) + 2 * Uold(i, j, k, ID) + Uold(i, j, k+1, ID)) * grav;
            // Thermal source term
            rhoc = Unew(i, j, k, ID);
            uc = Unew(i, j, k, IU) / rhoc;
            vc = Unew(i, j, k, IV) / rhoc;
            wc = Unew(i, j, k, IW) / rhoc;
            ekinc = 0.5 * (uc * uc + vc * vc + wc * wc) * rhoc;
            egc = rhoc * Unew(i, j, k, IG);
            Tc = (Unew(i, j, k, IE) - ekinc - egc) / (cv * rhoc);
            Teq = T_bottom + dTdz * (zc[k] - zc[1]);
            Tnew = (Tc + Teq * dt / tau) / (1.0 + dt / tau);
            Unew(i, j, k, IE) = rhoc * cv * Tnew + ekinc + egc;
        });
}

// This function generates a linearly spaced array of points between a specified start and stop value.
// It takes the start and stop values, the number of points to generate, and an output array to hold the results.
// The first and last points in the output array are set to the start and stop values respectively,
// and the intermediate points are calculated using linear interpolation.
void linspace(double start, double end, int num, Kokkos::View<double*>& out) {
    double step = (end - start) / (num - 1);
    Kokkos::parallel_for("linspace", num, KOKKOS_LAMBDA(int i) {
        out(i) = start + i * step;
    });
}

// This program simulates a fluid dynamics problem using a computational grid.
// It initializes parameters, allocates memory for data structures, computes initial conditions,
// applies boundary conditions, and iteratively solves the hydrodynamic equations over a specified number of time steps.
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc,argv);
    {
    using namespace conv_variables;

    // Declare variables
    int it, nt, freq_output;  // Time step index
    double elapsed_time;  // Timing variables
    double cfl, dt;

    // Simulation parameters
    nx = 100;  // Number of grid cells in x-direction
    ny = 2;    // Number of grid cells in y-direction
    nz = 50;   // Number of grid cells in z-direction
    nt = 3000; // Total number of time steps to simulate
    cfl = 0.45; // CFL condition for stability
    freq_output = 150; // Frequency of output writing (every 150 time steps)
    Lx = 2.0;  // Length of the domain in x-direction
    Ly = 1.0;  // Length of the domain in y-direction
    Lz = 1.0;  // Length of the domain in z-direction
    gam = 1.01; // Specific heat ratio for the fluid
    cv = 5.0;  // Heat capacity of the fluid
    grav = -1.0; // Gravitational acceleration (negative indicates downward)

    // Temperature forcing parameters
    tau = 1.0; // Time scale for temperature forcing
    T_bottom = 10.0; // Temperature at the bottom of the domain
    rho_bottom = 10.0; // Density at the bottom of the domain
    dTdz = -5.0; // Temperature gradient in the z-direction

    // Grid spacing calculations
    dx = Lx / nx; // Grid spacing in x-direction
    dy = Ly / ny; // Grid spacing in y-direction
    dz = Lz / nz; // Grid spacing in z-direction

    // Instantiate Kokkos view for grid coordinates
    // Arrays for x, y, and z cell center coordinates with ghosts
    Kokkos::View<double*> xc("xc", nx + 2), yc("yc", ny + 2), zc("zc", nz + 2);

    // Allocate memory for grid coordinates
    linspace(-0.5 * dx, Lx + 0.5 * dx, nx + 2, xc); // Generate x-coordinates
    linspace(-0.5 * dy, Ly + 0.5 * dy, ny + 2, yc); // Generate y-coordinates
    linspace(-0.5 * dz, Lz + 0.5 * dz, nz + 2, zc); // Generate z-coordinates

    // Allocate memory for simulation data structures
    Kokkos::View<double****> Uold("Uold", nx + 2, ny + 2, nz + 2, nvar);
    Kokkos::View<double****> Unew("Unew", nx + 2, ny + 2, nz + 2, nvar);

    // Compute initial conditions for the simulation
    compute_initial_condition(Unew, xc, yc, zc);

    // Copy initial conditions from Unew to Uold and apply boundary conditions
    Kokkos::deep_copy(Uold, Unew);
    compute_boundary_condition(Uold, xc, yc, zc);

    // Creates a host mirror view for output (when using GPUs)
    auto Uhost = Kokkos::create_mirror_view(Uold);
    Kokkos::deep_copy(Uhost, Uold);
    auto xc_host = Kokkos::create_mirror_view(xc), yc_host = Kokkos::create_mirror_view(yc), zc_host = Kokkos::create_mirror_view(zc);
    Kokkos::deep_copy(xc_host,xc), Kokkos::deep_copy(yc_host,yc), Kokkos::deep_copy(zc_host,zc);

    // Get the start time for performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Time-stepping loop
    for (it = 0; it < nt; ++it) {
        // Write output for the current time step
        write_output(it, freq_output, Uhost, xc_host, yc_host, zc_host);

        // Compute the time step for the next iteration
        dt = cfl * compute_timestep(Uold);

        // Solve the hydrodynamic equations using the kernel
        compute_kernel(Uold, Unew, dt, xc, yc, zc);

        // Update Uold with the new values from Unew and apply boundary conditions
        Kokkos::deep_copy(Uold, Unew);
        compute_boundary_condition(Uold, xc, yc, zc);
        Kokkos::deep_copy(Uhost, Uold);
    }

    // Measure elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Execution time (s): " << elapsed_time << '\n';
    std::cout << "Performance (Mcell-update/s): " << nt * nx * ny * nz/ (1E6 * elapsed_time) << '\n';
    }
    Kokkos::finalize();
    return 0;
}
