 module conv_variables
    integer, parameter :: nvar = 6  ! Number of variables (e.g., density, momentum, energy, etc.)
    integer :: ID, IU, IV, IW, IE, IG   ! Indices for different variables
    integer :: nx, ny, nz, nt, freq_output  ! Grid dimensions and simulation parameters
    real(8) :: cfl, Lx, Ly, Lz, gamma, cv, grav  ! Physical constants and parameters
    real(8) :: tau, T_bottom, rho_bottom, dTdz, dx, dy, dz, dt  ! Additional parameters for temperature and grid spacing
    real(8), dimension(:), allocatable :: xc, yc, zc  ! Arrays for x, y, and z cell center coordinates with ghosts
    real(8), dimension(:,:,:,:), allocatable :: Uold, Unew  ! 3D arrays for old and new state variables
end module conv_variables

! This program simulates a fluid dynamics problem using a computational grid.
! It initializes parameters, allocates memory for data structures, computes initial conditions,
! applies boundary conditions, and iteratively solves the hydrodynamic equations over a specified number of time steps.
program main
  use conv_variables
  implicit none

  ! Declare variables
  integer :: it                        ! Time step index
  real(8) :: start_time, end_time, elapsed_time  ! Timing variables

  ! Define indices for data structure
  ID = 1                   ! Input data identifier for density
  IU = 2                   ! Output data identifier for x-momentum
  IV = 3                   ! Variable identifier for y-momentum
  IW = 4                   ! Variable identifier for z-momentum
  IE = 5                   ! Variable identifier for total energy
  IG = 6                   ! Variable identifier for gravitational potential energy

  ! Simulation parameters
  nx = 100                 ! Number of grid cells in x-direction
  ny = 1                   ! Number of grid cells in y-direction
  nz = 50                  ! Number of grid cells in z-direction
  nt = 3000                ! Total number of time steps to simulate
  cfl = 0.45               ! CFL condition for stability
  freq_output = 150        ! Frequency of output writing (every 150 time steps)
  Lx = 2.0                 ! Length of the domain in x-direction
  Ly = 1.0                 ! Length of the domain in y-direction
  Lz = 1.0                 ! Length of the domain in z-direction
  gamma = 1.01             ! Specific heat ratio for the fluid
  cv = 5.0                 ! Heat capacity of the fluid
  grav = -1.0              ! Gravitational acceleration (negative indicates downward)

  ! Temperature forcing parameters
  tau = 1.0                ! Time scale for temperature forcing
  T_bottom = 10.0          ! Temperature at the bottom of the domain
  rho_bottom = 10.0        ! Density at the bottom of the domain
  dTdz = -5.0              ! Temperature gradient in the z-direction

  ! Grid spacing calculations
  dx = Lx / nx             ! Grid spacing in x-direction
  dy = Ly / ny             ! Grid spacing in y-direction
  dz = Lz / nz             ! Grid spacing in z-direction

  ! Allocate memory for grid coordinates
  allocate(xc(0:nx+1), yc(0:ny+1), zc(0:nz+1))  ! Allocate arrays for x, y, and z coordinates
  call linspace(-0.5 * dx, Lx + 0.5 * dx, nx + 2, xc)  ! Generate x-coordinates
  call linspace(-0.5 * dy, Ly + 0.5 * dy, ny + 2, yc)  ! Generate y-coordinates
  call linspace(-0.5 * dz, Lz + 0.5 * dz, nz + 2, zc)  ! Generate z-coordinates

  ! Allocate memory for simulation data structures
  allocate(Uold(0:nx+1, 0:ny+1, 0:nz+1, nvar), Unew(0:nx+1, 0:ny+1, 0:nz+1, nvar))  ! Allocate arrays for simulation state

  ! Compute initial conditions for the simulation
  call compute_initial_condition

  ! Copy initial conditions from Unew to Uold and apply boundary conditions
  Uold = Unew
  call compute_boundary_condition

  ! Get the start time for performance measurement
  call cpu_time(start_time)

  ! Time-stepping loop
  do it = 0, nt - 1
      ! Write output for the current time step
      call write_output(it)

      ! Compute the time step for the next iteration
      call compute_timestep

      ! Solve the hydrodynamic equations using the kernel
      call compute_kernel

      ! Update Uold with the new values from Unew and apply boundary conditions
      Uold = Unew
      call compute_boundary_condition
  end do

  ! Get the end time after the simulation
  call cpu_time(end_time)

  ! Calculate and display the elapsed time and performance metrics
  elapsed_time = end_time - start_time
  write(*,*) 'Execution time (s): ', elapsed_time
  write(*,*) 'Performance (Mcell-update/s): ', nt * nx * ny * nz/ (1E6 * elapsed_time)  ! Calculate performance in million cell updates per second

   contains

! This subroutine fills an output data array with the state variables of the fluid simulation
! at each grid cell. It computes the density, velocity components, kinetic energy, gravitational energy,
! and temperature for each cell in the grid and stores these values in the provided data array.
subroutine fill_data(data)
    use conv_variables
    implicit none

    ! Input/Output parameter
    real(8), intent(inout) :: data(nx*ny*nz, nvar + 3)  ! Output data array

    ! Loop variables
    integer :: i, j, k, index

    ! Temporary variables for calculations
    real(8) :: rhoc, uc, vc, wc, ekinc, egc, Tc

    ! Loop over the grid cells in the z-direction
    do k = 1, nz
        ! Loop over the grid cells in the y-direction
        do j = 1, ny
            ! Loop over the grid cells in the x-direction
            do i = 1, nx
                ! Retrieve density and velocity components
                rhoc = Uold(i, j, k, ID)              ! Density
                uc = Uold(i, j, k, IU) / rhoc         ! X-velocity
                vc = Uold(i, j, k, IV) / rhoc         ! Y-velocity
                wc = Uold(i, j, k, IW) / rhoc         ! Z-velocity

                ! Calculate kinetic and gravitational energy
                ekinc = 0.5 * (uc**2 + vc**2 + wc**2) * rhoc  ! Kinetic energy
                egc = rhoc * Uold(i, j, k, IG)          ! Gravitational energy

                ! Calculate temperature
                Tc = (Uold(i, j, k, IE) - ekinc - egc) / (cv * rhoc)

                ! Fill the data array with computed values
                index = nx * ny * (k - 1) + nx * (j - 1) + i
                data(index, 1) = xc(i)  ! X-coordinate
                data(index, 2) = yc(j)  ! Y-coordinate
                data(index, 3) = zc(k)  ! Z-coordinate
                data(index, 4) = Uold(i, j, k, ID)  ! Density
                data(index, 5) = Uold(i, j, k, IU)  ! X-momentum
                data(index, 6) = Uold(i, j, k, IV)  ! Y-momentum
                data(index, 7) = Uold(i, j, k, IW)  ! Z-momentum
                data(index, 8) = Uold(i, j, k, IE)  ! Internal energy
                data(index, 9) = Tc              ! Temperature
            end do
        end do
    end do
end subroutine fill_data

! This subroutine writes the output of the fluid simulation to a CSV file at specified time steps.
! It calculates the maximum kinetic energy across the grid, prints the current time step and kinetic energy,
! and saves the simulation state to a file if the output frequency condition is met.
subroutine write_output(it)
    use conv_variables
    implicit none

    integer, intent(in) :: it                     ! Current time step
    real(8) :: ekin_max                            ! Maximum kinetic energy
    integer :: iout, i                             ! Output index and loop variable
    real(8), dimension(:,:), allocatable :: data   ! Array to hold data for output
    character(len=100) :: output_file              ! Output file name

    ! Calculate the maximum kinetic energy across the grid
    ekin_max = maxval(0.5 * (Uold(1:nx, 1:ny, 1:nz, IU)**2 + Uold(1:nx, 1:ny, 1:nz, IV)**2 + Uold(1:nx, 1:ny, 1:nz, IW)**2) / Uold(1:nx, 1:ny, 1:nz, ID))

    ! Print the current time step and kinetic energy to the console
    print *, "timestep: ", it, "kinetic energy: ", ekin_max

    ! Check if it's time to write output based on the frequency
    if (mod(it, freq_output) == 0) then
        iout = it / freq_output                       ! Calculate output index
        write(*,*) "write csv output: ", iout       ! Inform about the output being generated

        allocate(data(nx*ny*nz, nvar + 3))

        ! Fill the data array with the current simulation state
        call fill_data(data)

        ! Generate the output file name
        write(output_file, '("output.csv.", I6.6)') iout

        ! Open the output file for writing
        open(unit=10, file=output_file, status='replace')

        ! Write the header for the CSV file
        write(10, '(A)') "x,y,z,rho,rhou,rhov,rhow,rhoE,temperature"

        ! Write the data to the output file
        do i = 1, nx * ny * nz
            write(10, '(F10.5, 8(",", F10.5))') &
                data(i, 1), data(i, 2), data(i, 3), &
                data(i, 4), data(i, 5), data(i, 6), &
                data(i, 7), data(i, 8), data(i, 9)
        end do

        ! Close the output file
        close(10)
    end if
end subroutine write_output

! This subroutine initializes the conditions for a fluid simulation in a 2D grid.
! It sets the initial values for density, momentum, total energy, and gravitational potential energy
! at each grid cell based on specified boundary conditions and temperature gradients.
subroutine compute_initial_condition
    use conv_variables
    implicit none

    ! Loop variables
    integer :: i, j, k
    ! Temporary variables for temperature and density calculations
    real(8) :: Tl, Tr, rhol, rhor, ur, vr, wr

    ! Initialize the bottom of the domain at k=1
    do i = 1, nx
        do j = 1, ny
            Unew(i, j, 1, ID) = rho_bottom  ! Set density at the bottom of the domain
            Unew(i, j, 1, IU) = 0.0          ! Set initial X-momentum at the bottom to zero
            Unew(i, j, 1, IV) = 0.0          ! Set initial Y-momentum at the bottom to zero
            Unew(i, j, 1, IW) = 0.0          ! Set initial Z-momentum at the bottom to zero
            Unew(i, j, 1, IE) = rho_bottom * cv * T_bottom + rho_bottom * (-grav * zc(1))  ! Total energy at the bottom, considering gravitational potential
            Unew(i, j, 1, IG) = -grav * zc(1)  ! Gravitational potential energy at the bottom
        end do
    end do

    ! Recursive initialization of the rest of the domain
    do i = 1, nx
        do j = 1, ny
            do k = 2, nz
                ! Calculate left and right temperatures based on the vertical position
                Tl = T_bottom + dTdz * (zc(k-1) - zc(1))  ! Temperature at the previous grid cell (k-1)
                Tr = T_bottom + dTdz * (zc(k) - zc(1))    ! Temperature at the current grid cell (k)

                rhol = Unew(i, j, k-1, ID)  ! Retrieve density from the previous grid cell

                ! Calculate the density at the current grid cell using the temperature gradient
                rhor = rhol * ((gamma - 1.0) * cv * Tl + 0.5 * grav * dz) / &
                          ((gamma - 1.0) * cv * Tr - 0.5 * grav * dz)

                ur = 0.0  ! Set initial x-velocity to zero
                vr = 0.0  ! Set initial y-velocity to zero

                ! Generate a small random value for z-velocity
                call random_number(wr)
                wr = 1E-3 * wr  ! Scale the random value

                ! Fill the state variables for the current grid cell
                Unew(i, j, k, ID) = rhor  ! Density
                Unew(i, j, k, IU) = rhor * ur  ! X-momentum
                Unew(i, j, k, IV) = rhor * vr  ! Y-momentum
                Unew(i, j, k, IW) = rhor * wr  ! Z-momentum
                Unew(i, j, k, IE) = rhor * cv * Tr + rhor * (-grav * zc(k)) + 0.5 * (ur**2 + vr**2 + wr**2) * rhor  ! Total energy
                Unew(i, j, k, IG) = -grav * zc(k)  ! Gravitational potential energy
            end do
        end do
    end do
end subroutine compute_initial_condition

! This subroutine computes boundary conditions for a fluid simulation in a 2D grid.
! It extrapolates values for the bottom and top boundaries based on the values from the
! two adjacent layers, and sets periodic boundary conditions in the x/y-directions.
! The calculations involve density, velocity, kinetic energy, internal energy, and
! temperature, taking into account gravitational effects.
subroutine compute_boundary_condition
    use conv_variables
    implicit none
    ! Loop variables
    integer :: i, j, k
    ! Temporary variables for calculations
    real(8) :: rho2, u2, v2, w2, ekin2, eg2, T2
    real(8) :: rho1, u1, v1, w1, ekin1, eg1, T1
    real(8) :: T0, rho0, ekin0, eg0
    ! Extrapolation for the bottom boundary (k = 0)
    do i = 1, nx
        do j = 1, ny
            ! Get the values from the second layer
            rho2 = Uold(i, j, 2, ID)
            u2 = Uold(i, j, 2, IU) / rho2
            v2 = Uold(i, j, 2, IV) / rho2
            w2 = Uold(i, j, 2, IW) / rho2
            ekin2 = 0.5 * (u2**2 + v2**2 + w2**2) * rho2
            eg2 = rho2 * Uold(i, j, 2, IG)
            T2 = (Uold(i, j, 2, IE) - ekin2 - eg2) / (cv * rho2)
            ! Get the values from the first layer
            rho1 = Uold(i, j, 1, ID)
            u1 = Uold(i, j, 1, IU) / rho1
            v1 = Uold(i, j, 1, IV) / rho1
            w1 = Uold(i, j, 1, IW) / rho1
            ekin1 = 0.5 * (u1**2 + v1**2 + w1**2) * rho1
            eg1 = rho1 * Uold(i, j, 1, IG)
            T1 = (Uold(i, j, 1, IE) - ekin1 - eg1) / (cv * rho1)
            ! Calculate the extrapolated values
            T0 = 2.0 * T1 - T2
            rho0 = rho1 * ((gamma - 1.0) * cv * T1 - 0.5 * grav * dz) / &
                        ((gamma - 1.0) * cv * T0 + 0.5 * grav * dz)
            ! Set boundary conditions for the bottom layer
            Uold(i, j, 0, ID) = rho0
            Uold(i, j, 0, IU) = rho0 * u1
            Uold(i, j, 0, IV) = rho0 * v1
            Uold(i, j, 0, IW) = -rho0 * w1
            ekin0 = ekin1 / rho1 * rho0
            eg0 = rho0 * (-grav * zc(1))
            Uold(i, j, 0, IE) = rho0 * cv * T0 + ekin0 + eg0
            Uold(i, j, 0, IG) = -grav * zc(1)
        end do
    end do
    ! Extrapolation for the top boundary (k = nz + 1)
    do i = 1, nx
        do j = 1, ny
            rho2 = Uold(i, j, nz-1, ID)
            u2 = Uold(i, j, nz-1, IU) / rho2
            v2 = Uold(i, j, nz-1, IV) / rho2
            w2 = Uold(i, j, nz-1, IV) / rho2
            ekin2 = 0.5 * (u2**2 + v2**2 + w2**2) * rho2
            eg2 = rho2 * Uold(i, j, nz-1, IG)
            T2 = (Uold(i, j, nz-1, IE) - ekin2 - eg2) / (cv * rho2)

            rho1 = Uold(i, j, nz, ID)
            u1 = Uold(i, j, nz, IU) / rho1
            v1 = Uold(i, j, nz, IV) / rho1
            w1 = Uold(i, j, nz, IW) / rho1
            ekin1 = 0.5 * (u1**2 + v1**2 + w1**2) * rho1
            eg1 = rho1 * Uold(i, j, nz, IG)
            T1 = (Uold(i, j, nz, IE) - ekin1 - eg1) / (cv * rho1)
            T0 = 2.0 * T1 - T2
            rho0 = rho1 * ((gamma - 1.0) * cv * T1 + 0.5 * grav * dz) / &
                        ((gamma - 1.0) * cv * T0 - 0.5 * grav * dz)
            ! Set boundary conditions for the top layer
            Uold(i, j, nz + 1, ID) = rho0
            Uold(i, j, nz + 1, IU) = rho0 * u1
            Uold(i, j, nz + 1, IV) = rho0 * v1
            Uold(i, j, nz + 1, IW) = -rho0 * w1
            ekin0 = ekin1 / rho1 * rho0
            eg0 = rho0 * (-grav * zc(nz + 1))
            Uold(i, j, nz + 1, IE) = rho0 * cv * T0 + ekin0 + eg0
            Uold(i, j, nz + 1, IG) = -grav * zc(nz + 1)
        end do
    end do

    ! Periodic boundary conditions in the x-direction
    do k = 0, nz + 1
        do j = 1, ny
            Uold(0, j, k, :) = Uold(nx, j, k, :)      ! Left boundary
            Uold(nx + 1, j, k, :) = Uold(1, j, k, :)  ! Right boundary
        end do
    end do

    ! Periodic boundary conditions in the y-direction
    do k = 0, nz + 1
        do i = 0, nx + 1
            Uold(i, 0, k, :) = Uold(i, ny, k, :)      ! Left boundary
            Uold(i, ny + 1, k, :) = Uold(i, 1, k, :)  ! Right boundary
        end do
    end do
end subroutine compute_boundary_condition

! This subroutine computes the local time step for a numerical simulation
! based on the CFL condition. It iterates over a grid defined by nx and ny,
! calculating the density, velocities, kinetic and gravitational energy,
! pressure, and speed of sound at each grid cell. The global time step
! is updated to the minimum value found across all grid cells to ensure
! stability in the simulation.
subroutine compute_timestep
    use conv_variables
    implicit none
    ! Local variables for time step calculation
    real(8) :: dt_loc, rhoc, uc, vc, wc, ekinc, egc, pc, ac
    integer :: i, j, k
    dt = 1.0E20  ! Initialize the time step to a large value
    ! Loop over the grid cells to compute the local time step
    do i = 1, nx
        do j = 1, ny
            do k = 1, nz
                ! Calculate the density and velocity components
                rhoc = Uold(i, j, k, ID)
                uc = Uold(i, j, k, IU) / rhoc  ! X-velocity
                vc = Uold(i, j, k, IV) / rhoc  ! Y-velocity
                wc = Uold(i, j, k, IW) / rhoc  ! Z-velocity
                ! Calculate kinetic and gravitational energy
                ekinc = 0.5 * (uc**2 + vc**2 + wc**2) * rhoc
                egc = rhoc * Uold(i, j, k, IG)
                ! Calculate pressure and sound speed
                pc = (Uold(i, j, k, IE) - ekinc - egc) * (gamma - 1.0)
                ac = sqrt(gamma * pc / rhoc)  ! Speed of sound
                ! Calculate the local time step based on CFL condition
                dt_loc = cfl * min(min(dx, dy), dz) / (ac + sqrt(uc**2 + vc**2 + wc**2))
                ! Update the global time step to the minimum found
                dt = min(dt, dt_loc)
            end do
        end do
    end do
end subroutine compute_timestep


! This subroutine computes the flux at a face between two states in a convection problem.
! It takes in left and right state vectors, gravitational acceleration, and outputs the flux vector.
! The calculations involve determining various properties (density, velocity, energy) for both states,
! as well as face-centered properties, and then applying a numerical flux method based on wave speeds.
subroutine compute_flux(Ucl, Ucr, gravdx, flux)
    use conv_variables
    implicit none
    ! Input parameters
    real(8), dimension(nvar) :: Ucl, Ucr, flux  ! Left and right state vectors, and output flux
    real(8) :: gravdx                          ! Gravitational acceleration in the x-direction
    ! Local variables for left state calculations
    real(8) :: rhol, ul, vl, wl, ekinl, egl, pl, al
    ! Local variables for right state calculations
    real(8) :: rhor, ur, vr, wr, ekinr, egr, pr, ar
    ! Additional variables for flux calculations
    real(8) :: aface, ustar, theta, pstar
    ! Calculate left state properties
    rhol = Ucl(ID)
    ul = Ucl(IU) / rhol
    vl = Ucl(IV) / rhol
    wl = Ucl(IW) / rhol
    ekinl = 0.5 * (ul**2 + vl**2 + wl**2) * rhol
    egl = rhol * Ucl(IG)
    pl = (Ucl(IE) - ekinl - egl) * (gamma - 1.0)
    al = rhol * sqrt(gamma * pl / rhol)
    ! Calculate right state properties
    rhor = Ucr(ID)
    ur = Ucr(IU) / rhor
    vr = Ucr(IV) / rhor
    wr = Ucr(IW) / rhor
    ekinr = 0.5 * (ur**2 + vr**2 + wr**2) * rhor
    egr = rhor * Ucr(IG)
    pr = (Ucr(IE) - ekinr - egr) * (gamma - 1.0)
    ar = rhor * sqrt(gamma * pr / rhor)
    ! Calculate face-centered properties
    aface = 1.1 * max(al, ar)  ! Face speed
    ustar = 0.5 * (ul + ur) - 0.5 * (pr - pl - 0.5 * (rhol + rhor) * gravdx) / aface
    theta = min(abs(ustar) / max(al / rhol, ar / rhor), 1.0)  ! Non-dimensional speed
    pstar = 0.5 * (pl + pr) - 0.5 * (ur - ul) * aface * theta  ! Star pressure
    ! Calculate fluxes based on the wave speed (ustar)
    if (ustar > 0.0) then
        flux(ID) = ustar * Ucl(ID)
        flux(IU) = ustar * Ucl(IU) + pstar
        flux(IV) = ustar * Ucl(IV)
        flux(IW) = ustar * Ucl(IW)
        flux(IE) = ustar * Ucl(IE) + pstar * ustar
    else
        flux(ID) = ustar * Ucr(ID)
        flux(IU) = ustar * Ucr(IU) + pstar
        flux(IV) = ustar * Ucr(IV)
        flux(IW) = ustar * Ucr(IW)
        flux(IE) = ustar * Ucr(IE) + pstar * ustar
    end if
end subroutine compute_flux


! This Fortran subroutine computes the numerical fluxes for a two-dimensional grid
! using a finite volume method. It updates the state variables Unew based on the
! previous state Uold, considering both x and y directional fluxes, as well as
! gravitational and thermal source terms. The computations involve updating
! the energy and temperature based on fluid dynamics principles and the
! conservation equations.
subroutine compute_kernel
    use conv_variables
    implicit none
    ! Loop variables
    integer :: i, j, k, ivar
    ! State vectors for left and right states and computed flux
    real(8), dimension(nvar) :: Ul, Ur, flux
    ! Variables for energy and temperature calculations
    real(8) :: egc, ekinc, rhoc, tc, teq, tnew, uc, vc, wc
    ! Loop over the grid cells
    do i = 1, nx
        do j = 1, ny
            do k = 1, nz
                ! Compute fluxes in the x direction (left and right)
                ! Left flux in x direction
                Ul = Uold(i-1, j, k, :)
                Ur = Uold(i, j, k, :)
                flux = 0.0
                call compute_flux(Ul, Ur, 0.d0, flux)
                do ivar = 1, nvar
                    Unew(i, j, k, ivar) = Unew(i, j, k, ivar) + dt / dx * flux(ivar)
                end do
                ! Right flux in x direction
                Ul = Uold(i, j, k, :)
                Ur = Uold(i+1, j, k, :)
                call compute_flux(Ul, Ur, 0.d0, flux)
                do ivar = 1, nvar
                    Unew(i, j, k, ivar) = Unew(i, j, k, ivar) - dt / dx * flux(ivar)
                end do
                ! Compute fluxes in the y direction (up and down)
                ! Left flux in y direction
                Ul = Uold(i, j-1, k, :)
                Ur = Uold(i, j, k, :)
                call swap_direction(Ul, IV)
                call swap_direction(Ur, IV)
                call compute_flux(Ul, Ur, 0.d0, flux)
                call swap_direction(flux, IV)
                do ivar = 1, nvar
                    Unew(i, j, k, ivar) = Unew(i, j, k, ivar) + dt / dy * flux(ivar)
                end do
                ! Right flux in y direction
                Ul = Uold(i, j, k, :)
                Ur = Uold(i, j+1, k, :)
                call swap_direction(Ul, IV)
                call swap_direction(Ur, IV)
                call compute_flux(Ul, Ur, 0.d0, flux)
                call swap_direction(flux, IV)
                do ivar = 1, nvar
                    Unew(i, j, k, ivar) = Unew(i, j, k, ivar) - dt / dy * flux(ivar)
                end do
                ! Compute fluxes in the z direction (up and down)
                ! Left flux in z direction
                Ul = Uold(i, j, k-1, :)
                Ur = Uold(i, j, k, :)
                call swap_direction(Ul, IW)
                call swap_direction(Ur, IW)
                call compute_flux(Ul, Ur, grav * dz, flux)
                call swap_direction(flux, IW)
                do ivar = 1, nvar
                    Unew(i, j, k, ivar) = Unew(i, j, k, ivar) + dt / dz * flux(ivar)
                end do
                ! Right flux in y direction
                Ul = Uold(i, j, k, :)
                Ur = Uold(i, j, k+1, :)
                call swap_direction(Ul, IW)
                call swap_direction(Ur, IW)
                call compute_flux(Ul, Ur, grav * dz, flux)
                call swap_direction(flux, IW)
                do ivar = 1, nvar
                    Unew(i, j, k, ivar) = Unew(i, j, k, ivar) - dt / dz * flux(ivar)
                end do
                ! Gravity source term
                Unew(i, j, k, IW) = Unew(i, j, k, IW) + dt * 0.25 * (Uold(i, j, k-1, ID) + 2 * Uold(i, j, k, ID) + Uold(i, j, k+1, ID)) * grav
                ! Thermal source term
                rhoc = Unew(i, j, k, ID)
                uc = Unew(i, j, k, IU) / rhoc
                vc = Unew(i, j, k, IV) / rhoc
                wc = Unew(i, j, k, IW) / rhoc
                ekinc = 0.5 * (uc**2 + vc**2 + wc**2) * rhoc
                egc = rhoc * Unew(i, j, k, IG)
                Tc = (Unew(i, j, k, IE) - ekinc - egc) / (cv * rhoc)
                Teq = T_bottom + dTdz * (zc(k) - zc(1))
                Tnew = (Tc + Teq * dt / tau) / (1.0 + dt / tau)
                Unew(i, j, k, IE) = rhoc * cv * Tnew + ekinc + egc
            end do
        end do
    end do
end subroutine compute_kernel


! This subroutine swaps the x and y or z velocity components in a given array.
! It takes a one-dimensional array 'arr' as input/output and exchanges and the y/z direction
! the values at indices IU and IV/IW, which represent the x and y/z velocities respectively.
subroutine swap_direction(arr, IVW)
    implicit none
    integer :: IVW
    real(8), dimension(:), intent(inout) :: arr  ! Input/output array to swap velocities
    real(8) :: temp                               ! Temporary variable for swapping
    ! Swap the x and y velocity components
    temp = arr(IU)         ! Store the x-velocity in a temporary variable
    arr(IU) = arr(IVW)     ! Assign the y/z- velocity to the x-velocity position
    arr(IVW) = temp        ! Assign the stored x-velocity to the y/z-velocity position
end subroutine swap_direction

! This subroutine generates a linearly spaced array of points between a specified start and stop value.
! It takes the start and stop values, the number of points to generate, and an output array to hold the results.
! The first and last points in the output array are set to the start and stop values respectively,
! and the intermediate points are calculated using linear interpolation.
subroutine linspace(start, stop, num, output)
    implicit none
    real(8), intent(in) :: start, stop          ! Start and stop values for the range
    integer, intent(in) :: num                   ! Number of points to generate
    real(8), dimension(num), intent(out) :: output  ! Output array to hold the generated points
    integer :: i                                  ! Loop variable
    ! Set the first and last points in the output array
    output(1) = start
    output(num) = stop
    ! Generate points in the specified range
    do i = 2, num - 1
        output(i) = start + (stop - start) * (i - 1) / (num - 1)
    end do
end subroutine linspace

end program main
