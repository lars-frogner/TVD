import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation


def get_grid(n, mb, start, end):
    nb = n + 2*mb
    sb = 0
    s = mb
    e = s + n
    eb = e + mb
    d = (end - start)/(n - 1)
    x = np.array([d*(i-mb) for i in range(nb)])
    return d, x, s, e, sb, eb, nb


class System1D:

    def __init__(self, x_start=0, x_end=1, nx=100):

        self.nx = nx
        self.mb = 2
        self.x_start = x_start
        self.x_end = x_end
        self.dx, self.x, self.xs, self.xe, self.xsb, self.xeb, self.nxb = get_grid(nx, mb, x_start, x_end)
        self.x = self.x[np.newaxis, :]

    def set_periodic_BC(self, U):
        xs, xe, mb = self.get_interior_bounds(U)
        U[:, :xs] = U[:, xe-mb:xe]
        U[:, xe:] = U[:, xs:xs+mb]

    def set_reflective_BC(self, U):
        xs, xe, mb = self.get_interior_bounds(U)
        U[:, :xs] = U[:, xs+mb-1:xs-1:-1]
        U[:, xe:] = U[:, xe-1:xe-mb-1:-1]
        U[1, :xs] *= -1
        U[1, xe:] *= -1

    def set_fixed_BC(self, U_left, U_right, U):
        xs, xe, mb = self.get_interior_bounds(U)
        U[:, :xs] = U_left[:, np.newaxis]
        U[:, xe:] = U_right[:, np.newaxis]

    def set_continuous_BC(self, U):
        xs, xe, mb = self.get_interior_bounds(U)
        U[:, :xs] = U[:, xs:xs+1]
        U[:, xe:] = U[:, xe-1:xe]

    def get_interior_bounds(self, U):
        mb = (U.shape[-1] - self.nx)//2
        return mb, mb + self.nx, mb

    def one_up(self, U, pad=0):
        xs, xe, mb = self.get_interior_bounds(U)
        return U[:, xs-pad+1:xe+pad+1]

    def one_dn(self, U, pad=0):
        xs, xe, mb = self.get_interior_bounds(U)
        return U[:, xs-pad-1:xe+pad-1]

    def mid(self, U, pad=0):
        xs, xe, mb = self.get_interior_bounds(U)
        return U[:, xs-pad:xe+pad]

    def diff_up(self, U, pad=0):
        return self.one_up(U, pad=pad) - self.mid(U, pad=pad)

    def diff_dn(self, U, pad=0):
        return self.mid(U, pad=pad) - self.one_dn(U, pad=pad)

    def get_timestep(self, U, C):
        return C*self.dx/self.max_courant_number(U)


class AdvectionSystem1D(System1D):

    # d/dt(rho) + d/dx(rho*vx) = 0

    def __init__(self, **grid_kwargs):
        System1D.__init__(self, **grid_kwargs)
        self.U = np.zeros((1, self.nxb)) # [rho]
        self.vx = np.zeros((1, self.nxb))

    def density(self, U):
        return U[0:1, :]

    def compute_flux(self, U, F):
        vpad = (U.shape[1] - self.nx)//2
        F[0, :] = self.density(U)*self.mid(self.vx, pad=vpad)

    def flux(self, U):
        F = np.zeros(U.shape)
        self.compute_flux(U, F)
        return F

    def init_square_wave(self, x0=0, x1=0.2, rho0=0.5, rho1=2, v0=1):
        self.U[0, self.xs:self.xe] = self.square_wave_rho(0, x0, x1, rho0, rho1, v0)
        self.vx[:] = v0
        self.analytical_funcs = {'density': lambda t: self.square_wave_rho(t, x0, x1, rho0, rho1, v0)}
        self.max_courant_number = lambda U: v0
        self.set_BC = self.set_periodic_BC
        self.init_func = self.init_square_wave

    def init_semicircle_wave(self, x0=0.2, width=0.3, rho0=1, rho1=2, v0=1):
        self.U[0, self.xs:self.xe] = self.semicircle_wave_rho(0, x0, width, rho0, rho1, v0)
        self.vx[:] = v0
        self.analytical_funcs = {'density': lambda t: self.semicircle_wave_rho(t, x0, width, rho0, rho1, v0)}
        self.max_courant_number = lambda U: v0
        self.set_BC = self.set_periodic_BC
        self.init_func = self.init_semicircle_wave

    def square_wave_rho(self, t, x0, x1, rho0, rho1, v0):
        rho = np.ones((1, self.nx))*rho0
        rho[np.logical_and(self.mid(self.x) >= x0, self.mid(self.x) <= x1)] = rho1
        return np.roll(rho, int(v0*t/self.dx + 0.5))

    def semicircle_wave_rho(self, t, x0, width, rho0, rho1, v0):
        halfwidth = width/2
        x_offset = np.abs(self.mid(self.x) - x0)
        rho = np.ones((1, self.nx))*rho0
        rho[x_offset <= halfwidth] = rho0 + rho1*np.sqrt(1 - (x_offset[x_offset <= halfwidth]/halfwidth)**2)
        return np.roll(rho, int(v0*t/self.dx + 0.5))

    def get_var(self, var_name):
        vars_funcs = {'density': self.density}
        return vars_funcs[var_name](self.U)


class HDSystem1D(AdvectionSystem1D):

    # d/dt(rho) + d/dx(rho*vx) = 0
    # d/dt(rho*vx) + d/dx(rho*vx*vx + p) = 0
    # d/dt(e) + d/dx(e*vx + p*vx) = 0

    def __init__(self, adiabatic_index=1.4, **grid_kwargs):
        AdvectionSystem1D.__init__(self, **grid_kwargs)
        self.adiabatic_index = adiabatic_index
        self.U = np.zeros((3, self.nxb)) # [rho, rho*vx, e]

    def momentum_x(self, U):
        return U[1:2, :]

    def energy(self, U):
        return U[2:3, :]

    def velocity_x(self, U):
        return self.momentum_x(U)/self.density(U)

    def kinetic_energy(self, U):
        vx = self.velocity_x(U)
        return self.density(U)*vx*vx/2

    def thermal_pressure(self, U):
        return (self.energy(U) - self.kinetic_energy(U))*(self.adiabatic_index - 1)

    def total_pressure(self, U):
        return self.thermal_pressure(U)

    def compute_flux(self, U, F):
        vx = self.velocity_x(U)
        p = self.total_pressure(U)
        F[0, :] = self.density(U)*vx
        F[1, :] = self.momentum_x(U)*vx + p
        F[2, :] = (self.energy(U) + p)*vx

    def max_courant_number(self, U):
        return np.max(np.abs(self.velocity_x(U)) +
                      np.sqrt(self.adiabatic_index*self.thermal_pressure(U)/self.density(U)))

    def init_blast_waves(self, x0=0.1, x1=0.9, rho0=1, p0=1000, p1=0.01, p2=100, v0=0):
        self.U[0, self.xs:self.xe] = rho0
        self.U[1, self.xs:self.xe] = rho0*v0
        self.U[2, np.logical_and(self.x >= self.x_start, self.x < x0)[0, :]] = self.compute_total_energy(rho0, p0, v0)
        self.U[2, np.logical_and(self.x >= x0, self.x < x1)[0, :]] = self.compute_total_energy(rho0, p1, v0)
        self.U[2, np.logical_and(self.x >= x1, self.x <= self.x_end)[0, :]] = self.compute_total_energy(rho0, p2, v0)
        self.analytical_funcs = None
        self.set_BC = self.set_reflective_BC
        self.init_func = self.init_blast_waves

    def init_rarefaction_waves(self, x0=0.5, rho0=1, p0=0.4, v0=-2, v1=2):
        self.U[0, self.xs:self.xe] = rho0
        self.U[1, np.logical_and(self.x >= self.x_start, self.x < x0)[0, :]] = rho0*v0
        self.U[1, np.logical_and(self.x >= x0, self.x <= self.x_end)[0, :]] = rho0*v1
        self.U[2, np.logical_and(self.x >= self.x_start, self.x < x0)[0, :]] = self.compute_total_energy(rho0, p0, v0)
        self.U[2, np.logical_and(self.x >= x0, self.x <= self.x_end)[0, :]] = self.compute_total_energy(rho0, p0, v1)
        self.analytical_funcs = None
        self.set_BC = self.set_continuous_BC
        self.init_func = self.init_rarefaction_waves

    def compute_total_energy(self, density, pressure, speed):
        return pressure/(self.adiabatic_index - 1) + density*speed*speed/2

    def get_var(self, var_name):
        vars_funcs = {'density': self.density,
                      'momentum_x': self.momentum_x,
                      'energy': self.energy,
                      'velocity_x': self.velocity_x,
                      'pressure': self.thermal_pressure}
        return vars_funcs[var_name](self.U)


class MHDSystem1D(HDSystem1D):

    # 2D:
    # d/dt(rho) + d/dx(rho*vx) = 0
    # d/dt(rho*vx) + d/dx(rho*vx*vx + p_tot - Bx*Bx/mu) = 0
    # d/dt(rho*vy) + d/dx(rho*vy*vx - Bx*By/mu) = 0
    # d/dt(e) + d/dx(e*vx + p_tot*vx - Bx*Bx*vx/mu - Bx*By*vy/mu) = 0
    # d/dt(Bx) = 0
    # d/dt(By) + d/dx(By*vx - Bx*vy) = 0

    # p_tot = p + (Bx*Bx + By*By)/(2*mu)
    # e = p/(gamma - 1) + rho*(vx*vx + vy*vy)/2 + (Bx*Bx + By*By)/(2*mu)

    # 3D:
    # d/dt(rho) + d/dx(rho*vx) = 0
    # d/dt(rho*vx) + d/dx(rho*vx*vx + p_tot - Bx*Bx/mu) = 0
    # d/dt(rho*vy) + d/dx(rho*vy*vx - Bx*By/mu) = 0
    # d/dt(rho*vz) + d/dx(rho*vz*vx - Bx*Bz/mu) = 0
    # d/dt(e) + d/dx(e*vx + p_tot*vx - Bx*Bx*vx/mu - Bx*By*vy/mu - Bx*Bz*vz/mu) = 0
    # d/dt(Bx) = 0
    # d/dt(By) + d/dx(By*vx - Bx*vy) = 0
    # d/dt(By) + d/dx(Bz*vx - Bx*vz) = 0

    # p_tot = p + (Bx*Bx + By*By + Bz*Bz)/(2*mu)
    # e = p/(gamma - 1) + rho*(vx*vx + vy*vy + vz*vz)/2 + (Bx*Bx + By*By + Bz*Bz)/(2*mu)

    def __init__(self, permeability=1, **HD_kwargs):
        HDSystem1D.__init__(self, **HD_kwargs)
        self.permeability = permeability
        self.U = np.zeros((8, self.nxb)) # [rho, rho*vx, rho*vy, rho*vz, e, Bx, By, Bz]

    def momentum_y(self, U):
        return U[2:3, :]

    def momentum_z(self, U):
        return U[3:4, :]

    def energy(self, U):
        return U[4:5, :]

    def B_field_x(self, U):
        return U[5:6, :]

    def B_field_y(self, U):
        return U[6:7, :]

    def B_field_z(self, U):
        return U[7:8, :]

    def velocity_y(self, U):
        return self.momentum_y(U)/self.density(U)

    def velocity_z(self, U):
        return self.momentum_z(U)/self.density(U)

    def kinetic_energy(self, U):
        vx = self.velocity_x(U)
        vy = self.velocity_y(U)
        vz = self.velocity_z(U)
        return self.density(U)*(vx*vx + vy*vy + vz*vz)/2

    def magnetic_pressure(self, U):
        Bx = self.B_field_x(U)
        By = self.B_field_y(U)
        Bz = self.B_field_z(U)
        return (Bx*Bx + By*By + Bz*Bz)/(2*self.permeability)

    def thermal_pressure(self, U):
        return (self.energy(U) - self.kinetic_energy(U) - self.magnetic_pressure(U))*(self.adiabatic_index - 1)

    def total_pressure(self, U):
        return self.thermal_pressure(U) + self.magnetic_pressure(U)

    def compute_flux(self, U, F):

        vx = self.velocity_x(U)
        vy = self.velocity_y(U)
        vz = self.velocity_z(U)
        Bx = self.B_field_x(U)
        By = self.B_field_y(U)
        Bz = self.B_field_z(U)
        p_tot = self.total_pressure(U)

        F[0, :] = self.density(U)*vx
        F[1, :] = self.momentum_x(U)*vx - Bx*Bx/self.permeability + p_tot
        F[2, :] = self.momentum_y(U)*vx - Bx*By/self.permeability
        F[3, :] = self.momentum_z(U)*vx - Bx*Bz/self.permeability
        F[4, :] = (self.energy(U) + p_tot)*vx - Bx*(Bx*vx + By*vy + Bz*vz)/self.permeability
        F[5, :] = 0
        F[6, :] = By*vx - Bx*vy
        F[7, :] = Bz*vx - Bx*vz

    def max_courant_number(self, U):
        flow_speeds_x = np.abs(self.velocity_x(U))
        flow_speeds_y = np.abs(self.velocity_y(U))
        flow_speeds_z = np.abs(self.velocity_z(U))
        Bx = self.B_field_x(U)
        By = self.B_field_y(U)
        Bz = self.B_field_z(U)
        Bx2 = Bx*Bx
        By2 = By*By
        Bz2 = Bz*Bz
        B2 = Bx2 + By2 + Bz2
        alfven_speed_squared = B2/(self.permeability*self.density(U))
        sound_speed_squared = self.adiabatic_index*self.thermal_pressure(U)/self.density(U)
        squared_speed_sum = sound_speed_squared + alfven_speed_squared
        fast_mode_wave_speeds_x = np.sqrt((squared_speed_sum + np.sqrt(squared_speed_sum*squared_speed_sum - 4*sound_speed_squared*alfven_speed_squared*Bx2/B2))/2)
        fast_mode_wave_speeds_y = np.sqrt((squared_speed_sum + np.sqrt(squared_speed_sum*squared_speed_sum - 4*sound_speed_squared*alfven_speed_squared*By2/B2))/2)
        fast_mode_wave_speeds_z = np.sqrt((squared_speed_sum + np.sqrt(squared_speed_sum*squared_speed_sum - 4*sound_speed_squared*alfven_speed_squared*Bz2/B2))/2)
        return max(np.max(flow_speeds_x + fast_mode_wave_speeds_x),
                   max(np.max(flow_speeds_y + fast_mode_wave_speeds_y),
                       np.max(flow_speeds_z + fast_mode_wave_speeds_z)))

    def init_magnetic_shock_tube(self, x0=0.5, rho0=1, rho1=0.125, p0=1, p1=0.1, vx0=0, vy0=0, vz0=0, vx1=0, vy1=0, vz1=0, Bx0=0.75, By0=1, Bz0=0, Bx1=0.75, By1=-1, Bz1=0):
        left_part = np.logical_and(self.x >= self.x_start, self.x < x0)[0, :]
        right_part = np.logical_and(self.x >= x0, self.x <= self.x_end)[0, :]
        self.U[0, left_part] = rho0
        self.U[0, right_part] = rho1
        self.U[1, left_part] = rho0*vx0
        self.U[1, right_part] = rho1*vx1
        self.U[2, left_part] = rho0*vy0
        self.U[2, right_part] = rho1*vy1
        self.U[3, left_part] = rho0*vz0
        self.U[3, right_part] = rho1*vz1
        self.U[4, left_part] = self.compute_total_energy(rho0, p0, vx0, vy0, vz0, Bx0, By0, Bz0)
        self.U[4, right_part] = self.compute_total_energy(rho1, p1, vx1, vy1, vz1, Bx1, By1, Bz1)
        self.U[5, left_part] = Bx0
        self.U[5, right_part] = Bx1
        self.U[6, left_part] = By0
        self.U[6, right_part] = By1
        self.U[7, left_part] = Bz0
        self.U[7, right_part] = Bz1
        self.analytical_funcs = None
        self.set_BC = self.set_continuous_BC
        self.init_func = self.init_magnetic_shock_tube

    def init_shear_alfven_waves(self, x0=1, x1=2, rho0=1, rho1=1, p0=1e-9, p1=1e-9, vx0=0, vy0=0, vz0=0, vx1=0, vy1=0.001, vz1=0, Bx0=1, By0=0, Bz0=0, Bx1=1, By1=0, Bz1=0):
        middle_part = np.logical_and(self.x > x0, self.x <= x1)[0, :]
        outer_part = np.logical_not(middle_part)
        self.U[0, outer_part] = rho0
        self.U[0, middle_part] = rho1
        self.U[1, outer_part] = rho0*vx0
        self.U[1, middle_part] = rho1*vx1
        self.U[2, outer_part] = rho0*vy0
        self.U[2, middle_part] = rho1*vy1
        self.U[3, outer_part] = rho0*vz0
        self.U[3, middle_part] = rho1*vz1
        self.U[4, outer_part] = self.compute_total_energy(rho0, p0, vx0, vy0, vz0, Bx0, By0, Bz0)
        self.U[4, middle_part] = self.compute_total_energy(rho1, p1, vx1, vy1, vz1, Bx1, By1, Bz1)
        self.U[5, outer_part] = Bx0
        self.U[5, middle_part] = Bx1
        self.U[6, outer_part] = By0
        self.U[6, middle_part] = By1
        self.U[7, outer_part] = Bz0
        self.U[7, middle_part] = Bz1
        self.analytical_funcs = None
        self.set_BC = self.set_continuous_BC
        self.init_func = self.init_shear_alfven_waves

    def compute_total_energy(self, rho, p, vx, vy, vz, Bx, By, Bz):
        return p/(self.adiabatic_index - 1) + rho*(vx*vx + vy*vy + vz*vz)/2 + (Bx*Bx + By*By + Bz*Bz)/(2*self.permeability)

    def get_var(self, var_name):
        vars_funcs = {'density': self.density,
                      'momentum_x': self.momentum_x,
                      'momentum_y': self.momentum_y,
                      'momentum_z': self.momentum_z,
                      'energy': self.energy,
                      'B_field_x': self.B_field_x,
                      'B_field_y': self.B_field_y,
                      'B_field_z': self.B_field_z,
                      'velocity_x': self.velocity_x,
                      'velocity_y': self.velocity_y,
                      'velocity_z': self.velocity_z,
                      'kinetic_energy': self.kinetic_energy,
                      'magnetic_pressure': self.magnetic_pressure,
                      'thermal_pressure': self.thermal_pressure,
                      'total_pressure': self.total_pressure}
        return vars_funcs[var_name](self.U)


class TVDSolver:

    def __init__(self, system, slope_limiter='S'):
        self.s = system
        self.slope_limiter_name = slope_limiter
        self.slope_limiters = {'M': self.slope_limiter_minmod,
                               'U': self.slope_limiter_UMIST,
                               'W': self.slope_limiter_Woodward,
                               'S': self.slope_limiter_superbee}
        self.slope_limiter = self.slope_limiters[slope_limiter]

    def init(self, *args, **kwargs):
        self.t = 0
        self.s.init_func(*args, **kwargs)
        self.s.set_BC(self.s.U)

    def step(self, C):

        dt = self.s.get_timestep(self.s.U, C)

        self.s.U[:, self.s.xs:self.s.xe] += self.MUSCL_full_step(dt)
        self.s.set_BC(self.s.U)

        self.t += dt

        return dt

    def MUSCL_predictor_step(self, dU_limited, dt):
        F_up = self.s.flux(self.s.mid(self.s.U, pad=1) + dU_limited/2)
        F_dn = self.s.flux(self.s.mid(self.s.U, pad=1) - dU_limited/2)
        return -dt/(2*self.s.dx)*(F_up - F_dn)

    def MUSCL_full_step(self, dt):

        dU_up = self.s.diff_up(self.s.U, pad=1)
        dU_dn = self.s.diff_dn(self.s.U, pad=1)
        dU_limited_c = self.slope_limiter(dU_up, dU_dn)

        U_halfstep_c = self.s.mid(self.s.U, pad=1) + self.MUSCL_predictor_step(dU_limited_c, dt)

        self.s.set_BC(U_halfstep_c)

        U_L_up = self.s.mid(U_halfstep_c) + self.s.mid(dU_limited_c)/2
        U_R_up = self.s.one_up(U_halfstep_c) - self.s.one_up(dU_limited_c)/2
        U_L_dn = self.s.one_dn(U_halfstep_c) + self.s.one_dn(dU_limited_c)/2
        U_R_dn = self.s.mid(U_halfstep_c) - self.s.mid(dU_limited_c)/2

        F_LR_up = (self.s.flux(U_L_up) + self.s.flux(U_R_up))/2
        F_LR_dn = (self.s.flux(U_L_dn) + self.s.flux(U_R_dn))/2

        dU_transport_c = -dt/self.s.dx*(F_LR_up - F_LR_dn)

        phi_LR_up = self.dissipative_limiter_LF(U_L_up, U_R_up, dt)
        phi_LR_dn = self.dissipative_limiter_LF(U_L_dn, U_R_dn, dt)

        dU_corrector_c = (phi_LR_up - phi_LR_dn)/2

        return dU_transport_c + dU_corrector_c

    def slope_limiter_minmod(self, w_up, w_dn):
        return self.minmod(w_dn, w_up)

    def slope_limiter_UMIST(self, w_up, w_dn):
        return self.minmod(2*w_dn,
                           2*w_up,
                           w_dn/4 + 3*w_up/4,
                           3*w_dn/4 + w_up/4)

    def slope_limiter_Woodward(self, w_up, w_dn):
        return self.minmod(2*w_dn,
                           2*w_up,
                           w_dn/2 + w_up/2)

    def slope_limiter_superbee(self, w_up, w_dn):
        sgn = np.sign(w_up)
        abs_w_up = np.abs(w_up)
        sgn_w_dn = sgn*w_dn
        result = np.maximum(np.minimum(2*abs_w_up, sgn_w_dn),
                            np.minimum(abs_w_up, 2*sgn_w_dn))
        result[result < 0] = 0
        return sgn*result

    def minmod(self, *wi):
        n = len(wi)
        sgn = np.sign(wi[0])
        result = np.zeros((n, *(wi[0].shape)))
        result[0, :, :] = np.abs(wi[0])
        for i in range(1, n):
            result[i, :, :] = sgn*wi[i]
        result = np.min(result, axis=0)
        result[result < 0] = 0
        return sgn*result

    def dissipative_limiter_yee(self, U_left, U_right):
        return U_right - U_left

    def dissipative_limiter_LF(self, U_left, U_right, dt):
        U_LR = (U_left + U_right)/2
        return dt/self.s.dx*self.s.max_courant_number(U_LR)*self.dissipative_limiter_yee(U_left, U_right)


class Visualizer:

    def __init__(self, solver):
        self.system = solver.s
        self.solver = solver

    def animate(self, var_name, C,
                      vmin=None, vmax=None,
                      marker='.-',
                      title='',
                      T=1, video_time=6, fps=30,
                      savename=False):

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(var_name)
        ax.set_title('{} (limiter = {}, C = {:g})'.format(title, self.solver.slope_limiter_name, C))

        if vmin is not None and vmax is not None:
            ax.set_ylim(vmin, vmax)

        plt.tight_layout()

        self.solver.init()

        x = self.system.mid(self.system.x)[0, :]
        var = self.system.mid(self.system.get_var(var_name))[0, :]

        plot_analytical = self.system.analytical_funcs is not None
        if plot_analytical:
            var_analytical_func = self.system.analytical_funcs[var_name]

        numerical_line, = ax.plot(x, var, marker)
        time_text = ax.text(0.01, 0.99, '', ha='left', va='top', transform=ax.transAxes)
        variation_text = ax.text(0.5, 0.99, '', ha='center', va='top', transform=ax.transAxes)
        error_text = ax.text(0.99, 0.99, '', ha='right', va='top', transform=ax.transAxes)

        if plot_analytical:
            analytical_line, = ax.plot(x, var_analytical_func(0)[0, :], '-', alpha=0.5)
            lines = (numerical_line, analytical_line)
        else:
            lines = (numerical_line,)

        n_frames = int(np.ceil(video_time*fps))
        dt_out = T/(n_frames - 1)

        def init():
            return (*lines, time_text, variation_text, error_text)

        def update(i):

            if i > 0:

                dt_avg = 0
                n_substeps = 0
                while self.solver.t < dt_out*i:
                    dt_avg += self.solver.step(C)
                    n_substeps = n_substeps + 1
                if n_substeps == 0:
                    dt_avg = self.solver.step(C)
                else:
                    dt_avg /= n_substeps

                full_var = self.system.get_var(var_name)
                var = self.system.mid(full_var)
                variation = np.sum(np.abs(self.system.diff_up(full_var)))
                lines[0].set_ydata(var[0, :])
                time_text.set_text('dt = {:g}\nt = {:g}'.format(dt_avg, self.solver.t))
                variation_text.set_text('V = {:g}'.format(variation))

                if plot_analytical:
                    var_analytical = var_analytical_func(self.solver.t)
                    error = np.sum(np.abs(var - var_analytical))/self.system.nx
                    lines[1].set_ydata(var_analytical[0, :])
                    error_text.set_text('E = {:g}'.format(error))

            return (*lines, time_text, variation_text, error_text)

        anim = animation.FuncAnimation(fig, update, init_func=init, blit=True, frames=np.arange(n_frames))

        if savename:
            anim.save(savename, writer=animation.FFMpegWriter(fps=fps,
                                                              bitrate=3200,
                                                              extra_args=['-vcodec', 'libx264']))
        else:
            plt.show()


def test_square_wave(C=0.8, slope_limiter='W'):
    system = AdvectionSystem1D(nx=100)
    system.init_square_wave()
    solver = TVDSolver(system, slope_limiter=slope_limiter)
    visualizer = Visualizer(solver)
    visualizer.animate('density', C, T=1, video_time=8,
                       title='Square wave',
                       savename='square_wave_rho_{}_{:g}.mp4'.format(slope_limiter, C))


def test_semicircle_wave(C=0.8, slope_limiter='W'):
    system = AdvectionSystem1D(nx=100)
    system.init_semicircle_wave()
    solver = TVDSolver(system, slope_limiter=slope_limiter)
    visualizer = Visualizer(solver)
    visualizer.animate('density', C, T=1, video_time=8,
                       title='Semicircle wave',
                       savename='semicircle_wave_rho_{}_{:g}.mp4'.format(slope_limiter, C))


def test_blast_waves(C=0.8, slope_limiter='W'):
    T = 0.076
    video_time = 8
    title = 'Two blast waves'
    system = HDSystem1D(nx=200)
    system.init_blast_waves()
    solver = TVDSolver(system, slope_limiter=slope_limiter)
    visualizer = Visualizer(solver)
    visualizer.animate('density', C, vmin=0, vmax=7, T=T, video_time=video_time, marker='.-',
                       title=title,
                       savename='blast_waves_rho_{}_{:g}.mp4'.format(slope_limiter, C))
    visualizer.animate('velocity_x', C, vmin=-2, vmax=17, T=T, video_time=video_time, marker='.-',
                       title=title,
                       savename='blast_waves_vx_{}_{:g}.mp4'.format(slope_limiter, C))
    system_hires = HDSystem1D(nx=1200)
    system_hires.init_blast_waves()
    solver_hires = TVDSolver(system_hires, slope_limiter=slope_limiter)
    visualizer_hires = Visualizer(solver_hires)
    visualizer_hires.animate('density', C, vmin=0, vmax=7, T=T, video_time=video_time, marker='-',
                             title=title,
                             savename='blast_waves_rho_hires_{}_{:g}.mp4'.format(slope_limiter, C))
    visualizer_hires.animate('velocity_x', C, vmin=-2, vmax=17, T=T, video_time=video_time, marker='-',
                             title=title,
                             savename='blast_waves_vx_hires_{}_{:g}.mp4'.format(slope_limiter, C))


def test_rarefaction_waves(C=0.4, slope_limiter='M'):
    T = 0.2
    video_time = 8
    title = 'Strong rarefaction waves'
    system = HDSystem1D(nx=100)
    system.init_rarefaction_waves()
    solver = TVDSolver(system, slope_limiter=slope_limiter)
    visualizer = Visualizer(solver)
    visualizer.animate('density', C, vmin=-0.1, vmax=1.1, T=T, video_time=video_time, marker='.-',
                       title=title,
                       savename='rarefaction_waves_rho_{}_{:g}.mp4'.format(slope_limiter, C))
    visualizer.animate('velocity_x', C, vmin=-2.1, vmax=2.1, T=T, video_time=video_time, marker='.-',
                       title=title,
                       savename='rarefaction_waves_vx_{}_{:g}.mp4'.format(slope_limiter, C))


def test_magnetic_shock_tube(C=0.8, slope_limiter='W'):
    T = 0.2
    video_time = 8
    title = 'Magnetic shock tube'
    system = MHDSystem1D(nx=200, adiabatic_index=2)
    system.init_magnetic_shock_tube()
    solver = TVDSolver(system, slope_limiter=slope_limiter)
    visualizer = Visualizer(solver)
    visualizer.animate('density', C, vmin=0, vmax=1.1, T=T, video_time=video_time, marker='.-',
                       title=title,
                       savename='magnetic_shock_tube_rho_{}_{:g}.mp4'.format(slope_limiter, C))
    visualizer.animate('velocity_x', C, vmin=-0.3, vmax=0.7, T=T, video_time=video_time, marker='.-',
                       title=title,
                       savename='magnetic_shock_tube_vx_{}_{:g}.mp4'.format(slope_limiter, C))
    system_hires = MHDSystem1D(nx=800, adiabatic_index=2)
    system_hires.init_magnetic_shock_tube()
    solver_hires = TVDSolver(system_hires, slope_limiter=slope_limiter)
    visualizer_hires = Visualizer(solver_hires)
    visualizer_hires.animate('density', C, vmin=0, vmax=1.1, T=T, video_time=video_time, marker='-',
                            title=title,
                            savename='magnetic_shock_tube_rho_hires_{}_{:g}.mp4'.format(slope_limiter, C))
    visualizer_hires.animate('velocity_x', C, vmin=-0.3, vmax=0.7, T=T, video_time=video_time, marker='-',
                            title=title,
                            savename='magnetic_shock_tube_vx_hires_{}_{:g}.mp4'.format(slope_limiter, C))


def test_shear_alfven_waves(C=0.8, slope_limiter='W'):
   T = 1.6
   video_time = 8
   title = 'Shear Alfven waves'
   system = MHDSystem1D(nx=150, x_end=3)
   system.init_shear_alfven_waves()
   solver = TVDSolver(system, slope_limiter=slope_limiter)
   visualizer = Visualizer(solver)
   visualizer.animate('velocity_y', C, vmin=-1e-4, vmax=7e-4, T=T, video_time=video_time, marker='.-',
                      title=title,
                      savename='shear_alfven_waves_vy_{}_{:g}.mp4'.format(slope_limiter, C))
   visualizer.animate('B_field_y', C, vmin=-6e-4, vmax=6e-4, T=T, video_time=video_time, marker='.-',
                      title=title,
                      savename='shear_alfven_waves_By_{}_{:g}.mp4'.format(slope_limiter, C))

if __name__ == '__main__':
    #test_square_wave()
    #test_semicircle_wave()
    #test_blast_waves()
    #test_rarefaction_waves()
    #test_magnetic_shock_tube()
    test_shear_alfven_waves()
