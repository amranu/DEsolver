import numpy as np

def initial_temperature(nx, ny):
    return np.zeros((nx, ny))

def initial_displacement(nx, ny):
    return np.zeros((nx, ny)), np.zeros((nx, ny))

def initial_potential_distribution(nx, ny):
    potential = np.zeros((nx, ny))
    potential[nx//4:3*nx//4, ny//4:3*ny//4] = 100  # Set the central region to a non-zero value
    return potential

def initial_source_distribution(nx, ny):
    source = np.zeros((nx, ny))
    source[nx//4:3*nx//4, ny//4:3*ny//4] = 10  # Set the central region to a non-zero value
    return source

def initial_wave_function(nx, ny):
    return np.zeros((nx, ny)), np.zeros((nx, ny))

def initial_gaussian_wave_packet(nx, ny, x0, y0, kx, ky, sigma):
    x = np.linspace(-nx/2, nx/2, nx)
    y = np.linspace(-ny/2, ny/2, ny)
    X, Y = np.meshgrid(x, y)
    psi_real = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2)) * np.cos(kx*X + ky*Y)
    psi_imag = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2)) * np.sin(kx*X + ky*Y)
    return psi_real, psi_imag

def initial_plane_wave(nx, ny, kx, ky):
    x = np.linspace(-nx/2, nx/2, nx)
    y = np.linspace(-ny/2, ny/2, ny)
    X, Y = np.meshgrid(x, y)
    psi_real = np.cos(kx*X + ky*Y)
    psi_imag = np.sin(kx*X + ky*Y)
    return psi_real, psi_imag

def initial_superposition_wave_packet(nx, ny, x0, y0, kx1, ky1, kx2, ky2, sigma):
    psi_real1, psi_imag1 = initial_gaussian_wave_packet(nx, ny, x0, y0, kx1, ky1, sigma)
    psi_real2, psi_imag2 = initial_gaussian_wave_packet(nx, ny, x0, y0, kx2, ky2, sigma)
    psi_real = psi_real1 + psi_real2
    psi_imag = psi_imag1 + psi_imag2
    return psi_real, psi_imag

def initial_double_gaussian_wave_packet(nx, ny, x1, y1, x2, y2, kx1, ky1, kx2, ky2, sigma):
    psi_real1, psi_imag1 = initial_gaussian_wave_packet(nx, ny, x1, y1, kx1, ky1, sigma)
    psi_real2, psi_imag2 = initial_gaussian_wave_packet(nx, ny, x2, y2, kx2, ky2, sigma)
    psi_real = psi_real1 + psi_real2
    psi_imag = psi_imag1 + psi_imag2
    return psi_real, psi_imag

def initial_velocity_vortex(nx, ny, strength=1.0):
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            x = i - nx // 2
            y = j - ny // 2
            r = np.sqrt(x**2 + y**2)
            if r != 0:
                u[i, j] = -strength * y / r
                v[i, j] = strength * x / r
    return u, v

def initial_velocity_shear_layer(nx, ny):
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if j < ny // 2:
                u[i, j] = 1.0
            else:
                u[i, j] = -1.0
    return u, v

def initial_velocity_random(nx, ny):
    u = np.random.rand(nx, ny) * 2 - 1
    v = np.random.rand(nx, ny) * 2 - 1
    return u, v

def initial_velocity_perturbation(nx, ny):
    u = np.ones((nx, ny))
    v = np.zeros((nx, ny))
    u[nx // 2, ny // 2] = 10.0  # Small perturbation
    return u, v

def initial_velocity_lid_driven_cavity(nx, ny):
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    u[:, -1] = 1.0  # Top lid moving to the right
    return u, v

def initial_pressure(nx, ny):
    return np.zeros((nx, ny))

def solve_heat_equation_step(temperature, alpha, dx, dy, dt):
    nx, ny = temperature.shape
    new_temperature = temperature.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_temperature[i, j] = temperature[i, j] + alpha * dt * (
                (temperature[i + 1, j] - 2 * temperature[i, j] + temperature[i - 1, j]) / dx**2 +
                (temperature[i, j + 1] - 2 * temperature[i, j] + temperature[i, j - 1]) / dy**2
            )
    return new_temperature

def solve_wave_equation_step(prev_disp, curr_disp, c, dx, dy, dt):
    nx, ny = curr_disp.shape
    new_disp = np.zeros_like(curr_disp)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_disp[i, j] = 2 * curr_disp[i, j] - prev_disp[i, j] + (c * dt)**2 * (
                (curr_disp[i + 1, j] - 2 * curr_disp[i, j] + curr_disp[i - 1, j]) / dx**2 +
                (curr_disp[i, j + 1] - 2 * curr_disp[i, j] + curr_disp[i, j - 1]) / dy**2
            )
    return curr_disp, new_disp

def solve_laplace_equation_step(potential, dx, dy):
    nx, ny = potential.shape
    new_potential = potential.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_potential[i, j] = 0.25 * (potential[i + 1, j] + potential[i - 1, j] + potential[i, j + 1] + potential[i, j - 1])
    return new_potential

def solve_poisson_equation_step(potential, source, dx, dy):
    nx, ny = potential.shape
    new_potential = potential.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_potential[i, j] = 0.25 * (potential[i + 1, j] + potential[i - 1, j] + potential[i, j + 1] + potential[i, j - 1] - dx * dy * source[i, j])
    return new_potential

def solve_schrodinger_equation_step(psi_real, psi_imag, V, dx, dy, dt, hbar, m):
    nx, ny = psi_real.shape
    new_psi_real = psi_real.copy()
    new_psi_imag = psi_imag.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_psi_real[i, j] += - (dt / (hbar * 2 * m)) * (
                - (hbar**2 / (2 * m)) * (
                    (psi_imag[i + 1, j] - 2 * psi_imag[i, j] + psi_imag[i - 1, j]) / dx**2 +
                    (psi_imag[i, j + 1] - 2 * psi_imag[i, j] + psi_imag[i, j - 1]) / dy**2
                ) + V[i, j] * psi_imag[i, j]
            )
            new_psi_imag[i, j] += (dt / (hbar * 2 * m)) * (
                - (hbar**2 / (2 * m)) * (
                    (psi_real[i + 1, j] - 2 * psi_real[i, j] + psi_real[i - 1, j]) / dx**2 +
                    (psi_real[i, j + 1] - 2 * psi_real[i, j] + psi_real[i, j - 1]) / dy**2
                ) + V[i, j] * psi_real[i, j]
            )
    return new_psi_real, new_psi_imag

def solve_navier_stokes_step(u, v, p, nu, dx, dy, dt, rho):
    nx, ny = u.shape
    new_u, new_v, new_p = u.copy(), v.copy(), p.copy()
    
    # Velocity field update
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_u[i, j] = u[i, j] + dt * (
                - u[i, j] * (u[i, j] - u[i - 1, j]) / dx
                - v[i, j] * (u[i, j] - u[i, j - 1]) / dy
                + nu * ((u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2)
                - (1 / rho) * (p[i + 1, j] - p[i - 1, j]) / (2 * dx)
            )
            new_v[i, j] = v[i, j] + dt * (
                - u[i, j] * (v[i, j] - v[i - 1, j]) / dx
                - v[i, j] * (v[i, j] - v[i, j - 1]) / dy
                + nu * ((v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2 + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy**2)
                - (1 / rho) * (p[i, j + 1] - p[i, j - 1]) / (2 * dy)
            )
    
    # Pressure field update
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            new_p[i, j] = p[i, j] - dt * rho * (
                (new_u[i + 1, j] - new_u[i - 1, j]) / (2 * dx)
                + (new_v[i, j + 1] - new_v[i, j - 1]) / (2 * dy)
            )
    
    return new_u, new_v, new_p

