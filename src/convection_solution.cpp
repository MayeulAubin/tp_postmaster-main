#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <Kokkos_Core.hpp>

using Array = Kokkos::View<double****>;
using ArrayHost = Array::HostMirror;
using ConstArray = Array::const_type;
using ConstArrayHost = ArrayHost::const_type;

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
}

struct Setup
{
    double gam, cv, grav;  // Physical constants and parameters
    double tau, T_bottom, rho_bottom, dTdz;  // Additional parameters for temperature and grid spacing
};

struct Mesh
{
    int nx, ny, nz;

    double Lx, Ly, Lz;

    double dx, dy, dz;

    Kokkos::View<double*> xc, yc, zc; // Arrays for x, y, and z cell center coordinates with ghosts

    Kokkos::View<double*>::HostMirror xc_host, yc_host, zc_host; // Arrays on the host
};

// This function fills an output data array with the state variables of the fluid simulation
// at each grid point. It computes the density, velocity components, kinetic energy, gravitational energy,
// and temperature for each point in the grid and stores these values in the provided data array.
void fill_data(Mesh const& mesh, Setup const& setup, std::vector<std::vector<double>>& data, ConstArrayHost const& U) {
    using namespace conv_variables;

    // Loop variables
    int i, j, k, index;

    // Temporary variables for calculations
    double rhoc, uc, vc, wc, ekinc, egc, Tc;

    // Loop over the grid points in the z-direction
    for (k = 1; k <= mesh.nz; ++k) {
        // Loop over the grid points in the y-direction
        for (j = 1; j <= mesh.ny; ++j) {
            // Loop over the grid points in the x-direction
            for (i = 1; i <= mesh.nx; ++i) {
                // Retrieve density and velocity components
                rhoc = U(i, j, k, ID);              // Density
                uc = U(i, j, k, IU) / rhoc;         // X-velocity
                vc = U(i, j, k, IV) / rhoc;         // Y-velocity
                wc = U(i, j, k, IW) / rhoc;         // Z-velocity

                // Calculate kinetic and gravitational energy
                ekinc = 0.5 * (uc * uc + vc * vc + wc * wc) * rhoc;  // Kinetic energy
                egc = rhoc * U(i, j, k, IG);          // Gravitational energy

                // Calculate temperature
                Tc = (U(i, j, k, IE) - ekinc - egc) / (setup.cv * rhoc);

                // Fill the data array with computed values
                index = mesh.nx * mesh.ny * (k - 1) + mesh.nx * (j - 1) + i -1;
                data[index][0] = mesh.xc_host(i);  // X-coordinate
                data[index][1] = mesh.yc_host(j);  // Y-coordinate
                data[index][2] = mesh.zc_host(k);  // Z-coordinate
                data[index][3] = U(i, j, k, ID);  // Density
                data[index][4] = U(i, j, k, IU);  // X-momentum
                data[index][5] = U(i, j, k, IV);  // Y-momentum
                data[index][6] = U(i, j, k, IW);  // Z-momentum
                data[index][7] = U(i, j, k, IE);  // Internal energy
                data[index][8] = Tc;              // Temperature
            }
        }
    }
}

// This function writes the output of the fluid simulation to a CSV file at specified time steps.
// It calculates the maximum kinetic energy across the grid, prints the current time step and kinetic energy,
// and saves the simulation state to a file if the output frequency condition is met.
void write_output(Mesh const& mesh, Setup const& setup, int it, int freq_output, ConstArray const& U) {
    using namespace conv_variables;

    double ekin_max; // Maximum ad local kinetic energy
    int iout; // Output index
    std::vector<std::vector<double>> data; // Array to hold data for output
    std::string output_file; // Output file name

    // Calculate the maximum kinetic energy across the grid
    Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {mesh.nx+1, mesh.ny+1, mesh.nz+1}),
            KOKKOS_LAMBDA(int i, int j, int k, double& ekin_loc)
            {
                double ekin_cell = 0.5 * (U(i, j, k, IU)*U(i, j, k, IU) + U(i, j, k, IV)*U(i, j, k, IV) + U(i, j, k, IW)*U(i, j, k, IW))/U(i, j, k, ID);
                ekin_loc = Kokkos::max(ekin_cell, ekin_loc);
            }, Kokkos::Max<double>(ekin_max));

    // Print the current time step and kinetic energy to the console
    std::cout << "timestep: " << it << " kinetic energy: " << ekin_max << '\n';

    // Check if it's time to write output based on the frequency
    if (it % freq_output == 0) {
        iout = it / freq_output; // Calculate output index
        std::cout << "write csv output: " << iout << '\n'; // Inform about the output being generated

	auto start = std::chrono::high_resolution_clock::now();
        data = std::vector<std::vector<double>>(mesh.nx * mesh.ny * mesh.nz, std::vector<double>(nvar + 3));
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "vector<vector<>> time (s): " << std::chrono::duration<double>(end - start).count() << '\n';	

	start = std::chrono::high_resolution_clock::now();
	auto uhost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "deep_copy time (s): " << std::chrono::duration<double>(end - start).count() << '\n';	

        // Fill the data array with the current simulation state
	start = std::chrono::high_resolution_clock::now();
        fill_data(mesh, setup, data, Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U));
	end = std::chrono::high_resolution_clock::now();
	std::cout << "fill_data time (s): " << std::chrono::duration<double>(end - start).count() << '\n';	

        // Generate the output file name
        output_file = "output.csv." + std::to_string(iout);

	start = std::chrono::high_resolution_clock::now();
        // Open the output file for writing
        std::ofstream outfile(output_file);

        // Write the header for the CSV file
        outfile << "x, y, z, rho, rhou, rhov, rhow, rhoE, temperature\n";

        // Write the data to the output file
	outfile << std::fixed << std::setprecision(5);
        for (int i = 0; i < mesh.nx * mesh.ny * mesh.nz; ++i) {
            outfile << data[i][0];
            for (int j = 1; j < 9; ++j) {
                outfile << ',' << data[i][j];
            }
            outfile << '\n';
        }

        // Close the output file
        outfile.close();
	end = std::chrono::high_resolution_clock::now();
	std::cout << "File dump time (s): " << std::chrono::duration<double>(end - start).count() << '\n';	
    }
}

// This function initializes the conditions for a fluid simulation in a 2D grid.
// It sets the initial values for density, momentum, total energy, and gravitational potential energy
// at each grid point based on specified boundary conditions and temperature gradients.
void compute_initial_condition(Mesh const& mesh, Setup const& setup, ArrayHost const& Unew) {
    using namespace conv_variables;

        // Loop variables
    int i, j, k;
    // Temporary variables for temperature and density calculations
    double Tl, Tr, rhol, rhor, ur, vr, wr;

    // Initialize the bottom of the domain at k=1
    for (i = 1; i <= mesh.nx; ++i) {
        for (j = 1; j <= mesh.ny; ++j) {
            Unew(i, j, 1, ID) = setup.rho_bottom;  // Set density at the bottom of the domain
            Unew(i, j, 1, IU) = 0.0;          // Set initial X-momentum at the bottom to zero
            Unew(i, j, 1, IV) = 0.0;          // Set initial Y-momentum at the bottom to zero
            Unew(i, j, 1, IW) = 0.0;          // Set initial Z-momentum at the bottom to zero
            Unew(i, j, 1, IE) = setup.rho_bottom * setup.cv * setup.T_bottom + setup.rho_bottom * (-setup.grav * mesh.zc_host(1));  // Total energy at the bottom, considering gravitational potential
            Unew(i, j, 1, IG) = -setup.grav * mesh.zc_host(1);  // Gravitational potential energy at the bottom
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // Recursive initialization of the rest of the domain
    for (i = 1; i <= mesh.nx; ++i) {
        for (j = 1; j <= mesh.ny; ++j) {
            for (k = 2; k <= mesh.nz; ++k) {
                // Calculate left and right temperatures based on the vertical position
                Tl = setup.T_bottom + setup.dTdz * (mesh.zc_host(k - 1) - mesh.zc_host(1));  // Temperature at the previous grid point (k-1)
                Tr = setup.T_bottom + setup.dTdz * (mesh.zc_host(k) - mesh.zc_host(1));      // Temperature at the current grid point (k)

                rhol = Unew(i, j, k - 1, ID);  // Retrieve density from the previous grid point

                // Calculate the density at the current grid point using the temperature gradient
                rhor = rhol * ((setup.gam - 1.0) * setup.cv * Tl + 0.5 * setup.grav * mesh.dz) /
                             ((setup.gam - 1.0) * setup.cv * Tr - 0.5 * setup.grav * mesh.dz);

                ur = 0.0;  // Set initial x-velocity to zero
                vr = 0.0;  // Set initial y-velocity to zero

                // Generate a small random value for z-velocity
                wr = 1E-3 * dis(gen);  // Scale the random value

                // Fill the state variables for the current grid point
                Unew(i, j, k, ID) = rhor;  // Density
                Unew(i, j, k, IU) = rhor * ur;  // X-momentum
                Unew(i, j, k, IV) = rhor * vr;  // Y-momentum
                Unew(i, j, k, IW) = rhor * wr;  // Z-momentum
                Unew(i, j, k, IE) = rhor * setup.cv * Tr + rhor * (-setup.grav * mesh.zc_host(k)) + 0.5 * (ur * ur + vr * vr + wr * wr) * rhor;  // Total energy
                Unew(i, j, k, IG) = -setup.grav * mesh.zc_host(k);  // Gravitational potential energy
            }
        }
    }
}

// This function computes boundary conditions for a fluid simulation in a 2D grid.
// It extrapolates values for the bottom and top boundaries based on the values from the
// two adjacent layers, and sets periodic boundary conditions in the x/y-directions.
// The calculations involve density, velocity, kinetic energy, internal energy, and
// temperature, taking into account gravitational effects.
void compute_boundary_condition(Mesh const& mesh, Setup const& setup, Array const& U) {
    using namespace conv_variables;

    // Extrapolation for the bottom boundary (k = 0)
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {mesh.nx+1, mesh.ny+1}),
        KOKKOS_LAMBDA(int i, int j)
        {
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
            T2 = (U(i, j, 2, IE) - ekin2 - eg2) / (setup.cv * rho2);

            // Get the values from the first layer
            rho1 = U(i, j, 1, ID);
            u1 = U(i, j, 1, IU) / rho1;
            v1 = U(i, j, 1, IV) / rho1;
            w1 = U(i, j, 1, IW) / rho1;
            ekin1 = 0.5 * (u1 * u1 + v1 * v1 + w1 * w1) * rho1;
            eg1 = rho1 * U(i, j, 1, IG);
            T1 = (U(i, j, 1, IE) - ekin1 - eg1) / (setup.cv * rho1);

            // Calculate the extrapolated values
            T0 = 2.0 * T1 - T2;
            rho0 = rho1 * ((setup.gam - 1.0) * setup.cv * T1 - 0.5 * setup.grav * mesh.dz) /
                        ((setup.gam - 1.0) * setup.cv * T0 + 0.5 * setup.grav * mesh.dz);

            // Set boundary conditions for the bottom layer
            U(i, j, 0, ID) = rho0;
            U(i, j, 0, IU) = rho0 * u1;
            U(i, j, 0, IV) = rho0 * v1;
            U(i, j, 0, IW) = -rho0 * w1;
            ekin0 = ekin1 / rho1 * rho0;
            eg0 = rho0 * (-setup.grav * mesh.zc(1));
            U(i, j, 0, IE) = rho0 * setup.cv * T0 + ekin0 + eg0;
            U(i, j, 0, IG) = -setup.grav * mesh.zc(1);
        });

    // Extrapolation for the top boundary (k = nz + 1)
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {mesh.nx+1, mesh.ny+1}),
        KOKKOS_LAMBDA(int i, int j)
        {
            // Temporary variables for calculations
            double rho2, u2, v2, w2, ekin2, eg2, T2;
            double rho1, u1, v1, w1, ekin1, eg1, T1;
            double T0, rho0, ekin0, eg0;

            rho2 = U(i, j, mesh.nz - 1, ID);
            u2 = U(i, j, mesh.nz - 1, IU) / rho2;
            v2 = U(i, j, mesh.nz - 1, IV) / rho2;
            w2 = U(i, j, mesh.nz - 1, IW) / rho2;
            ekin2 = 0.5 * (u2 * u2 + v2 * v2 + w2 * w2) * rho2;
            eg2 = rho2 * U(i, j, mesh.nz - 1, IG);
            T2 = (U(i, j, mesh.nz - 1, IE) - ekin2 - eg2) / (setup.cv * rho2);

            rho1 = U(i, j, mesh.nz, ID);
            u1 = U(i, j, mesh.nz, IU) / rho1;
            v1 = U(i, j, mesh.nz, IV) / rho1;
            w1 = U(i, j, mesh.nz, IW) / rho1;
            ekin1 = 0.5 * (u1 * u1 + v1 * v1 + w1 * w1) * rho1;
            eg1 = rho1 * U(i, j, mesh.nz, IG);
            T1 = (U(i, j, mesh.nz, IE) - ekin1 - eg1) / (setup.cv * rho1);

            T0 = 2.0 * T1 - T2;
            rho0 = rho1 * ((setup.gam - 1.0) * setup.cv * T1 + 0.5 * setup.grav * mesh.dz) /
                        ((setup.gam - 1.0) * setup.cv * T0 - 0.5 * setup.grav * mesh.dz);

            // Set boundary conditions for the top layer
            U(i, j, mesh.nz + 1, ID) = rho0;
            U(i, j, mesh.nz + 1, IU) = rho0 * u1;
            U(i, j, mesh.nz + 1, IV) = rho0 * v1;
            U(i, j, mesh.nz + 1, IW) = -rho0 * w1;
            ekin0 = ekin1 / rho1 * rho0;
            eg0 = rho0 * (-setup.grav * mesh.zc(mesh.nz + 1));
            U(i, j, mesh.nz + 1, IE) = rho0 * setup.cv * T0 + ekin0 + eg0;
            U(i, j, mesh.nz + 1, IG) = -setup.grav * mesh.zc(mesh.nz + 1);
        });

    // Periodic boundary conditions in the x-direction
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 1}, {mesh.nz+2, mesh.ny+1}),
        KOKKOS_LAMBDA(int k, int j)
        {
            for (int ivar = 0; ivar < nvar; ++ivar) {
                U(0, j, k, ivar) = U(mesh.nx, j, k, ivar);      // Left boundary
                U(mesh.nx + 1, j, k, ivar) = U(1, j, k, ivar);  // Right boundary
            }
        });

    // Periodic boundary conditions in the y-direction
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {mesh.nz+2, mesh.nx+2}),
        KOKKOS_LAMBDA(int k, int i)
        {
            for (int ivar = 0; ivar < nvar; ++ivar) {
                U(i, 0, k, ivar) = U(i, mesh.ny, k, ivar);      // Left boundary
                U(i, mesh.ny + 1, k, ivar) = U(i, 1, k, ivar);  // Right boundary
            }
        });
}

// This function computes the local time step for a numerical simulation
// based on the CFL condition. It iterates over a grid defined by nx and ny,
// calculating the density, velocities, kinetic and gravitational energy,
// pressure, and speed of sound at each grid point. The global time step
// is updated to the minimum value found across all grid points to ensure
// stability in the simulation.
double compute_timestep(Mesh const& mesh, Setup const& setup, ConstArray const& U) {
    using namespace conv_variables;
    double dt;

    // Loop over the grid points to compute the local time step
    Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {mesh.nx+1, mesh.ny+1, mesh.nz+1}),
            KOKKOS_LAMBDA(int i, int j, int k, double& dt_loc)
            {
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
                pc = (U(i, j, k, IE) - ekinc - egc) * (setup.gam - 1.0);
                ac = Kokkos::sqrt(setup.gam * pc / rhoc);  // Speed of sound

                // Calculate the local time step based on CFL condition
                // Update the global time step to the minimum found
                dt_loc = Kokkos::min(dt_loc, Kokkos::min(Kokkos::min(mesh.dx, mesh.dy), mesh.dz) / (ac + Kokkos::sqrt(uc * uc + vc * vc + wc * wc)));
            }, Kokkos::Min<double>(dt));

    return dt;
}

// This function computes the flux at a face between two states in a convection problem.
// It takes in left and right state vectors, gravitational acceleration, and outputs the flux vector.
// The calculations involve determining various properties (density, velocity, energy) for both states,
// as well as face-centered properties, and then applying a numerical flux method based on wave speeds.
KOKKOS_FUNCTION void compute_flux(Setup const& setup, Kokkos::Array<double, conv_variables::nvar> const& Ucl, Kokkos::Array<double, conv_variables::nvar> const& Ucr, double gravdx, Kokkos::Array<double, conv_variables::nvar>& flux) {
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
    pl = (Ucl[IE] - ekinl - egl) * (setup.gam - 1.0);
    al = rhol * Kokkos::sqrt(setup.gam * pl / rhol);

    // Calculate right state properties
    rhor = Ucr[ID];
    ur = Ucr[IU] / rhor;
    vr = Ucr[IV] / rhor;
    wr = Ucr[IW] / rhor;
    ekinr = 0.5 * (ur * ur + vr * vr + wr * wr) * rhor;
    egr = rhor * Ucr[IG];
    pr = (Ucr[IE] - ekinr - egr) * (setup.gam - 1.0);
    ar = rhor * Kokkos::sqrt(setup.gam * pr / rhor);

    // Calculate face-centered properties
    aface = 1.1 * Kokkos::max(al, ar);  // Face speed
    ustar = 0.5 * (ul + ur) - 0.5 * (pr - pl - 0.5 * (rhol + rhor) * gravdx) / aface;
    theta = Kokkos::min(Kokkos::fabs(ustar) / Kokkos::max(al / rhol, ar / rhor), 1.0);  // Non-dimensional speed
    pstar = 0.5 * (pl + pr) - 0.5 * (ur - ul) * aface * theta;  // Star pressure

    // Calculate fluxes based on the wave speed (ustar)
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
KOKKOS_FUNCTION void swap_direction(Kokkos::Array<double, conv_variables::nvar>& arr, int IVW) {
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
void compute_kernel(Mesh const& mesh, Setup const& setup, ConstArray const& Uold, Array& Unew, double dt) {
    using namespace conv_variables;

    // Loop over the grid points
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {mesh.nx+1, mesh.ny+1, mesh.nz+1}),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                // State vectors for left and right states and computed flux
                Kokkos::Array<double, nvar> Ul, Ur, flux;
                // Variables for energy and temperature calculations
                double egc, ekinc, rhoc, Tc, Teq, Tnew, uc, vc, wc;

                // Compute fluxes in the x direction (left and right)
                // Left flux in x direction
                for (int ivar = 0; ivar < nvar; ++ivar){
                    Ul[ivar] = Uold(i-1, j, k, ivar);
                    Ur[ivar] = Uold(i, j, k, ivar);
                }
                for (int ivar = 0; ivar < nvar; ++ivar){
                    flux[ivar] = 0.0;
                }
                compute_flux(setup, Ul, Ur, 0.0, flux);
                for (int ivar = 0; ivar < nvar; ++ivar) {
                    Unew(i, j, k, ivar) += dt / mesh.dx * flux[ivar];
                }
                // Right flux in x direction
                for (int ivar = 0; ivar < nvar; ++ivar){
                    Ul[ivar] = Uold(i, j, k, ivar);
                    Ur[ivar] = Uold(i+1, j, k, ivar);
                }
                compute_flux(setup, Ul, Ur, 0.0, flux);
                for (int ivar = 0; ivar < nvar; ++ivar) {
                    Unew(i, j, k, ivar) -= dt / mesh.dx * flux[ivar];
                }
                // Compute fluxes in the y direction (up and down)
                // Left flux in y direction
                for (int ivar = 0; ivar < nvar; ++ivar){
                    Ul[ivar] = Uold(i, j-1, k, ivar);
                    Ur[ivar] = Uold(i, j, k, ivar);
                }
                swap_direction(Ul, IV);
                swap_direction(Ur, IV);
                compute_flux(setup, Ul, Ur, 0.0, flux);
                swap_direction(flux, IV);
                for (int ivar = 0; ivar < nvar; ++ivar) {
                    Unew(i, j, k, ivar) += dt / mesh.dy * flux[ivar];
                }
                // Right flux in y direction
                for (int ivar = 0; ivar < nvar; ++ivar){
                    Ul[ivar] = Uold(i, j, k, ivar);
                    Ur[ivar] = Uold(i, j+1, k, ivar);
                }
                swap_direction(Ul, IV);
                swap_direction(Ur, IV);
                compute_flux(setup, Ul, Ur, 0.0, flux);
                swap_direction(flux, IV);
                for (int ivar = 0; ivar < nvar; ++ivar) {
                    Unew(i, j, k, ivar) -= dt / mesh.dy * flux[ivar];
                }
                // Compute fluxes in the z direction (up and down)
                // Left flux in z direction
                for (int ivar = 0; ivar < nvar; ++ivar){
                    Ul[ivar] = Uold(i, j, k-1, ivar);
                    Ur[ivar] = Uold(i, j, k, ivar);
                }
                swap_direction(Ul, IW);
                swap_direction(Ur, IW);
                compute_flux(setup, Ul, Ur, setup.grav * mesh.dz, flux);
                swap_direction(flux, IW);
                for (int ivar = 0; ivar < nvar; ++ivar) {
                    Unew(i, j, k, ivar) += dt / mesh.dz * flux[ivar];
                }
                // Right flux in z direction
                for (int ivar = 0; ivar < nvar; ++ivar){
                    Ul[ivar] = Uold(i, j, k, ivar);
                    Ur[ivar] = Uold(i, j, k+1, ivar);
                }
                swap_direction(Ul, IW);
                swap_direction(Ur, IW);
                compute_flux(setup, Ul, Ur, setup.grav * mesh.dz, flux);
                swap_direction(flux, IW);
                for (int ivar = 0; ivar < nvar; ++ivar) {
                    Unew(i, j, k, ivar) -= dt / mesh.dz * flux[ivar];
                }
                // Gravity source term
                Unew(i, j, k, IW) += dt * 0.25 * (Uold(i, j, k-1, ID) + 2 * Uold(i, j, k, ID) + Uold(i, j, k+1, ID)) * setup.grav;
                // Thermal source term
                rhoc = Unew(i, j, k, ID);
                uc = Unew(i, j, k, IU) / rhoc;
                vc = Unew(i, j, k, IV) / rhoc;
                wc = Unew(i, j, k, IW) / rhoc;
                ekinc = 0.5 * (uc * uc + vc * vc + wc * wc) * rhoc;
                egc = rhoc * Unew(i, j, k, IG);
                Tc = (Unew(i, j, k, IE) - ekinc - egc) / (setup.cv * rhoc);
                Teq = setup.T_bottom + setup.dTdz * (mesh.zc(k) - mesh.zc(1));
                Tnew = (Tc + Teq * dt / setup.tau) / (1.0 + dt / setup.tau);
                Unew(i, j, k, IE) = rhoc * setup.cv * Tnew + ekinc + egc;
            });
}

// This function generates a linearly spaced array of points between a specified start and stop value.
// It takes the start and stop values, the number of points to generate, and an output array to hold the results.
// The first and last points in the output array are set to the start and stop values respectively,
// and the intermediate points are calculated using linear interpolation.
void linspace(double start, double end, int num, Kokkos::View<double*> const& out) {
    double step = (end - start) / (num - 1);
    Kokkos::parallel_for(num, KOKKOS_LAMBDA(int i) {
        out(i) = start + i * step;
    });
}

// This program simulates a fluid dynamics problem using a computational grid.
// It initializes parameters, allocates memory for data structures, computes initial conditions,
// applies boundary conditions, and iteratively solves the hydrodynamic equations over a specified number of time steps.
int main() {
    using namespace conv_variables;

    Kokkos::ScopeGuard guard;

    // Declare variables
    int it, nt, freq_output;  // Time step index
    double elapsed_time;  // Timing variables
    double cfl, dt;
    Mesh mesh;
    Setup setup;

    // Simulation parameters
    mesh.nx = 200;  // Number of grid points in x-direction
    mesh.ny = 100;    // Number of grid points in y-direction
    mesh.nz = 100;   // Number of grid points in z-direction
    nt = 3000; // Total number of time steps to simulate
    cfl = 0.45; // CFL condition for stability
    freq_output = 150; // Frequency of output writing (every 150 time steps)
    mesh.Lx = 2.0;  // Length of the domain in x-direction
    mesh.Ly = 1.0;  // Length of the domain in y-direction
    mesh.Lz = 1.0;  // Length of the domain in z-direction
    setup.gam = 1.01; // Specific heat ratio for the fluid
    setup.cv = 5.0;  // Heat capacity of the fluid
    setup.grav = -1.0; // Gravitational acceleration (negative indicates downward)

    // Temperature forcing parameters
    setup.tau = 1.0; // Time scale for temperature forcing
    setup.T_bottom = 10.0; // Temperature at the bottom of the domain
    setup.rho_bottom = 10.0; // Density at the bottom of the domain
    setup.dTdz = -5.0; // Temperature gradient in the z-direction

    // Grid spacing calculations
    mesh.dx = mesh.Lx / mesh.nx; // Grid spacing in x-direction
    mesh.dy = mesh.Ly / mesh.ny; // Grid spacing in y-direction
    mesh.dz = mesh.Lz / mesh.nz; // Grid spacing in z-direction

    // Allocate memory for grid coordinates
    mesh.xc = Kokkos::View<double*>("xc", mesh.nx + 2);
    mesh.yc = Kokkos::View<double*>("yc", mesh.ny + 2);
    mesh.zc = Kokkos::View<double*>("zc", mesh.nz + 2);
    linspace(-0.5 * mesh.dx, mesh.Lx + 0.5 * mesh.dx, mesh.nx + 2, mesh.xc); // Generate x-coordinates
    linspace(-0.5 * mesh.dy, mesh.Ly + 0.5 * mesh.dy, mesh.ny + 2, mesh.yc); // Generate y-coordinates
    linspace(-0.5 * mesh.dz, mesh.Lz + 0.5 * mesh.dz, mesh.nz + 2, mesh.zc); // Generate z-coordinates
    mesh.xc_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.xc);
    mesh.yc_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.yc);
    mesh.zc_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.zc);

    // Allocate memory for simulation data structures
    Array Uold("Uold", mesh.nx + 2, mesh.ny + 2, mesh.nz + 2, nvar);
    Array Unew("Unew", mesh.nx + 2, mesh.ny + 2, mesh.nz + 2, nvar);

    // Compute initial conditions for the simulation
    ArrayHost Unew_host = Kokkos::create_mirror_view(Unew);
    compute_initial_condition(mesh, setup, Unew_host);
    Kokkos::deep_copy(Unew, Unew_host);

    // Copy initial conditions from Unew to Uold and apply boundary conditions
    Kokkos::deep_copy(Uold, Unew);
    compute_boundary_condition(mesh, setup, Uold);

    // Get the start time for performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Time-stepping loop
    for (it = 0; it < nt; ++it) {
        // Write output for the current time step
        write_output(mesh, setup, it, freq_output ,Uold);

        // Compute the time step for the next iteration
        dt = cfl * compute_timestep(mesh, setup, Uold);

        // Solve the hydrodynamic equations using the kernel
        compute_kernel(mesh, setup, Uold, Unew, dt);

        // Update Uold with the new values from Unew and apply boundary conditions
        Kokkos::deep_copy(Uold, Unew);
        compute_boundary_condition(mesh, setup, Uold);
    }

    // Measure elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Execution time (s): " << elapsed_time << '\n';
    std::cout << "Performance (Mcell-update/s): " << nt * mesh.nx * mesh.ny * mesh.nz/ (1E6 * elapsed_time) << '\n';

    return 0;
}
