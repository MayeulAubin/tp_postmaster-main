import numpy as np
from math import *
from numba import jit
import csv
import time

@jit(nopython=True)
def compute_initial_condition(U):
    """
    Computes the initial condition for the domain.

    Parameters:
    U (ndarray): The array representing the domain.

    Returns:
    None
    """
    # Initialize the bottom of the domain at k=1
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            # Set initial conditions for the first rows (k=1)
            U[i, j, 1, ID] = rho_bottom  # Set density
            U[i, j, 1, IU] = 0.0   # Set horizontal velocity
            U[i, j, 1, IV] = 0.0   # Set vertical velocity
            U[i, j, 1, IE] = rho_bottom * cv * T_bottom + rho_bottom * (-grav * zc[1])  # Set energy
            U[i, j, 1, IG] = -grav * zc[1]  # Set gravitational potential

    # Recursive initialization of the rest of the domain
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(2, nz + 1):
                # Calculate bottom and top temperatures based on vertical position
                Tl = T_bottom + dTdz * (zc[k - 1] - zc[1])  # Temperature at the bottom cell
                Tr = T_bottom + dTdz * (zc[k] - zc[1])      # Temperature at the top cell

                # Get the density from the previous row
                rhol = U[i, j, k - 1, ID]  # Density at the bottom cell

                # Calculate the density for the current cell
                rhor = rhol * ((gamma - 1.0) * cv * Tl + 0.5 * grav * dz) / \
                          ((gamma - 1.0) * cv * Tr - 0.5 * grav * dz)

                # Set velocities
                ur = 0.0  # x-velocity (remains zero)
                vr = 0.0  # y-velocity (remains zero)
                wr = 1E-3 * np.random.rand()  # z-velocity (small random value)

                # Assign computed values to the current cell
                U[i, j, k, ID] = rhor  # Density
                U[i, j, k, IU] = rhor * ur  # x-momentum
                U[i, j, k, IV] = rhor * vr  # y-momentum
                U[i, j, k, IW] = rhor * wr  # z-momentum
                U[i, j, k, IE] = rhor * cv * Tr + rhor * (-grav * zc[k]) + \
                            0.5 * (ur**2 + vr**2 + wr**2) * rhor  # Energy
                U[i, j, k, IG] = -grav * zc[k]  # Pressure


@jit(nopython=True)
def compute_boundary_condition(U):
    """
    Computes the boundary conditions for the domain.

    Parameters:
    U (ndarray): The array representing the domain.

    Returns:
    None
    """
    # Extrapolation for the bottom boundary (k=0)
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            # Retrieve values from the second row (k=2)
            rho2 = U[i,j, 2, ID]
            u2 = U[i, j, 2, IU] / rho2  # x-velocity
            v2 = U[i, j, 2, IV] / rho2  # y-velocity
            w2 = U[i, j, 2, IW] / rho2  # z-velocity
            ekin2 = 0.5 * (u2 ** 2 + v2 ** 2 + w2 **2) * rho2  # Kinetic energy
            eg2 = rho2 * U[i, j, 2, IG]  # Gravitational potential energy
            T2 = (U[i, j, 2, IE] - ekin2 - eg2) / (cv * rho2)  # Temperature from energy equation

            rho1 = U[i, j, 1, ID]
            u1 = U[i, j, 1, IU] / rho1  # x-velocity
            v1 = U[i, j, 1, IV] / rho1  # y-velocity
            w1 = U[i, j, 1, IW] / rho1  # z-velocity
            ekin1 = 0.5 * (u1 ** 2 + v1 ** 2 + w1 **2) * rho1  # Kinetic energy
            eg1 = rho1 * U[i, j, 1, IG]  # Gravitational potential energy
            T1 = (U[i, j, 1, IE] - ekin1 - eg1) / (cv * rho1)  # Temperature from energy equation

            # Calculate extrapolated values for the bottom boundary (j=0)
            T0 = 2 * T1 - T2  # Extrapolated temperature
            rho0 = rho1 * ((gamma - 1.0) * cv * T1 - 0.5 * grav * dz) / \
                        ((gamma - 1.0) * cv * T0 + 0.5 * grav * dz)  # Extrapolated density

            # Set values for the bottom boundary
            U[i, j, 0, ID] = rho0
            U[i, j, 0, IU] = rho0 * u1
            U[i, j, 0, IV] = rho0 * v1
            U[i, j, 0, IW] = -rho0 * w1  # Invert z-velocity for the bottom boundary
            ekin0 = ekin1 / rho1 * rho0  # Kinetic energy for the bottom boundary
            eg0 = rho0 * (-grav * zc[0])  # Gravitational potential energy for the bottom boundary
            U[i, j, 0, IE] = rho0 * cv * T0 + ekin0 + eg0  # Total energy
            U[i, j, 0, IG] = -grav * zc[0]  # Gravitational potential

    # Extrapolation for the top boundary (k=nz+1)
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            rho2 = U[i, j, nz - 1, ID]
            u2 = U[i, j, nz - 1, IU] / rho2  # x-velocity
            v2 = U[i, j, nz - 1, IV] / rho2  # y-velocity
            w2 = U[i, j, nz - 1, IW] / rho2  # z-velocity
            ekin2 = 0.5 * (u2 ** 2 + v2 ** 2 + w2 ** 2) * rho2  # Kinetic energy
            eg2 = rho2 * U[i, j, nz - 1, IG]  # Gravitational potential energy
            T2 = (U[i, j, nz - 1, IE] - ekin2 - eg2) / (cv * rho2)  # Temperature from energy equation

            rho1 = U[i, j, nz, ID]
            u1 = U[i, j, nz, IU] / rho1  # x-velocity
            v1 = U[i, j, nz, IV] / rho1  # y-velocity
            w1 = U[i, j, nz, IW] / rho1  # z-velocity
            ekin1 = 0.5 * (u1 ** 2 + v1 ** 2 + w1**2) * rho1  # Kinetic energy
            eg1 = rho1 * U[i, j, nz, IG]  # Gravitational potential energy
            T1 = (U[i, j, nz, IE] - ekin1 - eg1) / (cv * rho1)  # Temperature from energy equation

            # Calculate extrapolated values for the top boundary (k=nz+1)
            T0 = 2 * T1 - T2  # Extrapolated temperature
            rho0 = rho1 * ((gamma - 1.0) * cv * T1 + 0.5 * grav * dz) / \
                        ((gamma - 1.0) * cv * T0 - 0.5 * grav * dz)  # Extrapolated density

            # Set values for the top boundary
            U[i, j, nz + 1, ID] = rho0
            U[i, j, nz + 1, IU] = rho0 * u1
            U[i, j, nz + 1, IV] = rho0 * v1
            U[i, j, nz + 1, IW] = -rho0 * w1  # Invert z-velocity for the top boundary
            ekin0 = ekin1 / rho1 * rho0  # Kinetic energy for the top boundary
            eg0 = rho0 * (-grav * zc[nz + 1])  # Gravitational potential energy for the top boundary
            U[i, j, nz + 1, IE] = rho0 * cv * T0 + ekin0 + eg0  # Total energy
            U[i, j, nz + 1, IG] = -grav * zc[nz + 1]  # Gravitational potential

    # Apply periodic boundary conditions in the x-direction
    for k in range(nz + 2):
        for j in range(1, ny+1):
            U[0, j, k, :] = U[nx, j, k, :]  # Left boundary condition
            U[nx + 1, j, k, :] = U[1, j, k, :]  # Right boundary condition

    # Apply periodic boundary conditions in the y-direction
    for k in range(nz + 2):
        for i in range(nx+2):
            U[i, 0, k, :] = U[i, ny, k, :]  # Left boundary condition
            U[i, ny + 1, k, :] = U[i, 1, k, :]  # Right boundary condition


@jit(nopython=True)
def compute_timestep(U):
    """
    Computes the time step for the simulation.

    Parameters:
    U (ndarray): The array representing the domain.

    Returns:
    float: Computed time step.
    """
    # Initialize the time step to a very large value
    dt = 1E20

    # Loop over the grid cells in the x, y and z dimensions
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                # Get the density at the current grid cell
                rhoc = U[i, j, k, ID]

                # Calculate the velocities
                uc = U[i, j, k, IU] / rhoc  # x-velocity
                vc = U[i, j, k, IV] / rhoc  # y-velocity
                wc = U[i, j, k, IW] / rhoc  # z-velocity

                # Calculate kinetic energy and gravitational potential energy
                ekinc = 0.5 * (uc**2 + vc**2 + wc**2) * rhoc
                egc = rhoc * Uold[i, j, k,IG]  # Gravitational potential energy

                # Compute pressure using the energy equation
                pc = (U[i, j, k, IE] - ekinc - egc) * (gamma - 1.0)

                # Calculate the speed of sound
                ac = sqrt(gamma * pc / rhoc)

                # Compute the local time step based on the CFL condition
                dt_loc = min(min(dx, dy), dz)/ (ac + sqrt(uc**2 + vc**2 + wc**2))

                # Update the overall time step to the minimum found
                dt = min(dt, dt_loc)

    return dt

@jit(nopython=True)
def compute_flux(Ucl, Ucr, gravdx, flux):
    """
    Computes the flux between two states.

    Parameters:
    Ul (ndarray): Left state.
    Ur (ndarray): Right state.
    gravdx (float): Gravitational force * dx component in x-direction.
    flux (ndarray): Array to store the computed flux.

    Returns:
    None
    """
    # Extract properties from the left state
    rhol = Ucl[ID]  # Density
    ul = Ucl[IU] / rhol  # x-velocity
    vl = Ucl[IV] / rhol  # y-velocity
    wl = Ucl[IW] / rhol  # z-velocity
    ekinl = 0.5 * (ul**2 + vl**2 + wl**2) * rhol  # Kinetic energy
    egl = rhol * Ucl[IG]  # Gravitational potential energy
    pl = (Ucl[IE] - ekinl - egl) * (gamma - 1.0)  # Pressure
    al = rhol * sqrt(gamma * pl / rhol)  # acoustic impedance in left state

    # Extract properties from the right state
    rhor = Ucr[ID]  # Density
    ur = Ucr[IU] / rhor  # x-velocity
    vr = Ucr[IV] / rhor  # y-velocity
    wr = Ucr[IW] / rhor  # z-velocity
    ekinr = 0.5 * (ur**2 + vr**2 + wr**2) * rhor  # Kinetic energy
    egr = rhor * Ucr[IG]  # Gravitational potential energy
    pr = (Ucr[IE] - ekinr - egr) * (gamma - 1.0)  # Pressure
    ar = rhor * sqrt(gamma * pr / rhor)  # acoustic impedance in right state

    # Calculate face speed and pressure
    aface = 1.1 * max(al, ar)  # Face speed (slightly increased)
    ustar = 0.5 * (ul + ur) - 0.5 * (pr - pl - 0.5 * (rhol + rhor) * gravdx) / aface  # Star speed
    theta = min(abs(ustar) / max(al / rhol, ar / rhor), 1)  # Limiter for pressure calculation
    pstar = 0.5 * (pl + pr) - 0.5 * (ur - ul) * aface * theta  # Star pressure

    # Compute the flux based on the star speed
    if ustar > 0:
        flux[ID] = ustar * Ucl[ID]  # Density flux
        flux[IU] = ustar * Ucl[IU] + pstar  # x-momentum flux
        flux[IV] = ustar * Ucl[IV]  # y-momentum flux
        flux[IW] = ustar * Ucl[IW]  # z-momentum flux
        flux[IE] = ustar * Ucl[IE] + pstar * ustar  # Energy flux
    else:
        flux[ID] = ustar * Ucr[ID]  # Density flux
        flux[IU] = ustar * Ucr[IU] + pstar  # x-momentum flux
        flux[IV] = ustar * Ucr[IV]  # y-momentum flux
        flux[IW] = ustar * Ucr[IW]  # z-momentum flux
        flux[IE] = ustar * Ucr[IE] + pstar * ustar  # Energy flux


@jit(nopython=True)
def compute_kernel(Uold, Unew, dt):
    """
    Computes the kernel for updating the state of the system using flux calculations and source terms.

    Parameters:
    Uold (ndarray): The old state of the system.
    Unew (ndarray): The new state of the system to be updated.
    dt (float): Time step for the update.

    Returns:
    None
    """
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                # Initialize left and right states for flux calculations
                Ul = np.zeros(nvar)
                Ur = np.zeros(nvar)

                # x-direction left flux
                flux = np.zeros(nvar)
                Ul = Uold[i - 1, j, k, :].copy()  # Left state
                Ur = Uold[i, j, k, :].copy()      # Right state
                compute_flux(Ul, Ur, 0.0, flux)  # Compute flux
                for ivar in range(nvar):
                    Unew[i, j, k, ivar] += dt / dx * flux[ivar]  # Update new state

                # x-direction right flux
                flux.fill(0)  # Reset flux
                Ul = Uold[i, j, k, :].copy()  # Left state
                Ur = Uold[i + 1, j, k, :].copy()  # Right state
                compute_flux(Ul, Ur, 0.0, flux)  # Compute flux
                for ivar in range(nvar):
                    Unew[i, j, k, ivar] -= dt / dx * flux[ivar]  # Update new state

                # y-direction left flux
                flux.fill(0)  # Reset flux
                Ul = Uold[i, j - 1, k, :].copy()  # Left state
                Ur = Uold[i, j, k, :].copy()      # Right state
                # Swap direction for flux calculation
                Ul[IU], Ul[IV] = Ul[IV], Ul[IU]
                Ur[IU], Ur[IV] = Ur[IV], Ur[IU]
                compute_flux(Ul, Ur, 0.0, flux)  # Compute flux
                # Swap direction back
                flux[IU], flux[IV] = flux[IV], flux[IU]
                for ivar in range(nvar):
                    Unew[i, j, k, ivar] += dt / dy * flux[ivar]  # Update new state

                # y-direction right flux
                flux.fill(0)  # Reset flux
                Ul = Uold[i, j, k, :].copy()  # Left state
                Ur = Uold[i, j + 1, k, :].copy()  # Right state
                # Swap direction for flux calculation
                Ul[IU], Ul[IV] = Ul[IV], Ul[IU]
                Ur[IU], Ur[IV] = Ur[IV], Ur[IU]
                compute_flux(Ul, Ur, 0.0, flux)  # Compute flux
                # Swap direction back
                flux[IU], flux[IV] = flux[IV], flux[IU]
                for ivar in range(nvar):
                    Unew[i, j, k, ivar] -= dt / dy * flux[ivar]  # Update new state

                # z-direction left flux
                flux.fill(0)  # Reset flux
                Ul = Uold[i, j, k - 1, :].copy()  # Left state
                Ur = Uold[i, j, k, :].copy()      # Right state
                # Swap direction for flux calculation
                Ul[IU], Ul[IW] = Ul[IW], Ul[IU]
                Ur[IU], Ur[IW] = Ur[IW], Ur[IU]
                compute_flux(Ul, Ur, grav * dz, flux)  # Compute flux
                # Swap direction back
                flux[IU], flux[IW] = flux[IW], flux[IU]
                for ivar in range(nvar):
                    Unew[i, j, k, ivar] += dt / dz * flux[ivar]  # Update new state

                # z-direction right flux
                flux.fill(0)  # Reset flux
                Ul = Uold[i, j, k, :].copy()  # Left state
                Ur = Uold[i, j, k + 1, :].copy()  # Right state
                # Swap direction for flux calculation
                Ul[IU], Ul[IW] = Ul[IW], Ul[IU]
                Ur[IU], Ur[IW] = Ur[IW], Ur[IU]
                compute_flux(Ul, Ur, grav * dz, flux)  # Compute flux
                # Swap direction back
                flux[IU], flux[IW] = flux[IW], flux[IU]
                for ivar in range(nvar):
                    Unew[i, j, k, ivar] -= dt / dz * flux[ivar]  # Update new state


                # Gravity source term
                Unew[i, j, k, IW] += dt * 0.25 * (
                    Uold[i, j, k - 1, ID] + 2 * Uold[i, j, k, ID] + Uold[i, j, k + 1, ID]
                ) * grav

                # Thermal source term
                rhoc = Unew[i, j, k, ID]  # Density
                uc = Unew[i, j, k, IU] / rhoc  # x-velocity
                vc = Unew[i, j, k, IV] / rhoc  # y-velocity
                wc = Unew[i, j, k, IW] / rhoc  # z-velocity
                ekinc = 0.5 * (uc**2 + vc**2 + wc**2) * rhoc  # Kinetic energy
                egc = rhoc * Unew[i, j, k, IG]  # Gravitational potential energy
                Tc = (Unew[i, j, k, IE] - ekinc - egc) / (cv * rhoc)  # Temperature
                Teq = T_bottom + dTdz * (zc[k] - zc[1])  # Equilibrium temperature
                Tnew = (Tc + Teq * dt / tau) / (1.0 + dt / tau)  # New temperature
                Unew[i, j, k, IE] = rhoc * cv * Tnew + ekinc + egc  # Update energy


@jit(nopython=True)
def fill_data(data, U):
    """
    Fills the data array with values from the state array U.

    Parameters:
    data (ndarray): The array to store the filled data.
    U (ndarray): The state array from which to extract values.

    Returns:
    None
    """
    for k in range(1, nz + 1):  # Iterate over columns
        for j in range(1, ny + 1):  # Iterate over rows in j
            for i in range(1, nx + 1):  # Iterate over rows in i

                # Extract density and velocities
                rhoc = U[i, j, k, ID]
                uc = U[i, j, k, IU] / rhoc  # x-velocity
                vc = U[i, j, k, IV] / rhoc  # y-velocity
                wc = U[i, j, k, IW] / rhoc  # z-velocity

                # Calculate kinetic energy and gravitational potential energy
                ekinc = 0.5 * (uc**2 + vc**2 + wc**2) * rhoc
                egc = rhoc * U[i, j, k, IG]

                # Calculate temperature
                Tc = (U[i, j, k, IE] - ekinc - egc) / (cv * rhoc)

                # Fill the data array with coordinates and state variables
                index = nx * ny * (k -1) + nx * (j - 1) + i - 1
                data[index, 0] = xc[i]  # x-coordinate
                data[index, 1] = yc[j]  # y-coordinate
                data[index, 2] = zc[k]     # z-coordinate (set to 0)
                data[index, 3] = U[i, j, k, ID]  # Density
                data[index, 4] = U[i, j, k, IU]  # x-momentum
                data[index, 5] = U[i, j, k, IV]  # y-momentum
                data[index, 6] = U[i, j, k, IW]  # y-momentum
                data[index, 7] = U[i, j, k, IE]  # Energy
                data[index, 8] = Tc  # Temperature


def write_output(it, U):
    """
    Writes the output data to a CSV file.

    Parameters:
    it (int): The current timestep.
    U (ndarray): The state array from which to extract values.

    Returns:
    None
    """
    # Calculate maximum kinetic energy
    ekin_max = np.max(0.5 * (U[1:nx + 1, 1:ny + 1, 1:nz + 1, IU]**2 + U[1:nx + 1, 1:ny + 1, 1:nz + 1, IV]**2 + U[1:nx + 1, 1:ny + 1, 1:nz + 1, IW]**2) / U[1:nx + 1, 1:ny + 1, 1:nz + 1, ID])
    print("timestep: ", it, "kinetic energy: ", ekin_max)

    # Check if it's time to write output
    if it % freq_output == 0:
        iout = int(it / freq_output)
        print(f"write csv output: {iout}")

        # Prepare data for output
        data = np.zeros((nx * ny * nz, nvar + 3))
        fill_data(data, U)  # Fill the data array

        # Write data to CSV file
        output_file = f'output.csv.{iout:06d}'
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "z", "rho", "rhou", "rhov", "rhow", "rhoE", "temperature"])  # Write header
            writer.writerows(data)  # Write data rows


if __name__ == '__main__':
    # Constants for variable indices
    nvar = 6
    ID = 0  # Density
    IU = 1  # x-momentum
    IV = 2  # y-momentum
    IW = 3  # z-momentum
    IE = 4  # Energy
    IG = 5  # Gravitational potential

    # Simulation parameters
    nx = 100          # Number of grid cells in x-direction
    ny = 1            # Number of grid cells in y-direction
    nz = 50           # Number of grid cells in z-direction
    nt = 3000         # Number of time steps
    cfl = 0.45        # CFL condition
    freq_output = 150 # Frequency of output
    Lx = 2.0          # Length of the domain in x-direction
    Ly = 1.0          # Length of the domain in y-direction
    Lz = 1.0          # Length of the domain in z-direction
    gamma = 1.01      # Specific heat ratio
    cv = 5.0          # Specific heat at constant volume
    grav = -1.0       # Gravitational acceleration

    # Temperature forcing parameters
    tau = 1.0         # Time scale for temperature adjustment
    T_bottom = 10.0         # Base temperature
    rho_bottom = 10.0       # Base density
    dTdz = -5.0       # Temperature gradient in y-direction

    # Grid spacing
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    xc = np.linspace(-0.5 * dx, Lx + 0.5 * dx, nx + 2)  # x cell center coordinates with ghosts
    yc = np.linspace(-0.5 * dy, Ly + 0.5 * dy, ny + 2)  # y cell center coordinates with ghosts
    zc = np.linspace(-0.5 * dz, Lz + 0.5 * dz, nz + 2)  # z cell center coordinates with ghosts

    # Data structure initialization
    Uold = np.zeros((nx + 2, ny + 2, nz + 2, nvar))
    Unew = np.zeros((nx + 2, ny + 2, nz + 2, nvar))

    # Set initial conditions
    compute_initial_condition(Unew)

    # Copy initial conditions to Uold and apply boundary conditions
    Uold = Unew.copy()
    compute_boundary_condition(Uold)

    # Record the start time
    start_time = time.time()

    # Time loop for the simulation
    for it in range(nt):
        # Output data at specified intervals
        write_output(it, Uold)

        # Compute time step
        dt = cfl * compute_timestep(Uold)

        # Update the state using the hydro equations
        compute_kernel(Uold, Unew, dt)

        # Copy the new state to old state and apply boundary conditions
        Uold = Unew.copy()
        compute_boundary_condition(Uold)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time (s): {elapsed_time:.6f}")
    print(f"Performance: (Mcell-update/s): {nt * nx * ny * nz/ (elapsed_time * 1E6):.6f}")