import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation
import sys


def get_grid(n, mb, start, end):
    nb = n + 2*mb
    sb = 0
    s = mb
    e = s + n
    eb = e + mb
    d = (end - start)/(n - 1)
    x = np.array([d*(i-mb) for i in range(nb)])
    return d, x, s, e, sb, eb, nb


class System2D:

    def __init__(self, x_start=0, x_end=1, nx=100,
                       y_start=0, y_end=1, ny=100):

        self.mb = 2

        self.nx = nx
        self.x_start = x_start
        self.x_end = x_end
        self.dx, self.x, self.xs, self.xe, self.xsb, self.xeb, self.nxb = get_grid(self.nx, self.mb, self.x_start, self.x_end)
        self.x = self.x[np.newaxis, :]

        self.ny = ny
        self.y_start = y_start
        self.y_end = y_end
        self.dy, self.y, self.ys, self.ye, self.ysb, self.yeb, self.nyb = get_grid(self.ny, self.mb, self.y_start, self.y_end)
        self.y = self.y[np.newaxis, :]

    def set_periodic_BC_x(self, U):
        xs, xe, mbx = self.get_interior_bounds_x(U)
        U[:, :xs, :] = U[:, xe-mbx:xe, :]
        U[:, xe:, :] = U[:, xs:xs+mbx, :]

    def set_periodic_BC_y(self, U):
        ys, ye, mby = self.get_interior_bounds_y(U)
        U[:, :, :ys] = U[:, :, ye-mby:ye]
        U[:, :, ye:] = U[:, :, ys:ys+mby]

    def set_reflective_BC_x_bottom(self, U):
        xs, xe, mb = self.get_interior_bounds_x(U)
        U[:, :xs, :] = U[:, xs+mb-1:xs-1:-1, :]
        U[1, :xs, :] *= -1

    def set_reflective_BC_x_top(self, U):
        xs, xe, mb = self.get_interior_bounds_x(U)
        U[:, xe:, :] = U[:, xe-1:xe-mb-1:-1, :]
        U[1, xe:, :] *= -1

    def set_reflective_BC_y_bottom(self, U):
        ys, ye, mb = self.get_interior_bounds_y(U)
        U[:, :, :ys] = U[:, :, ys+mb-1:ys-1:-1]
        U[2, :, :ys] *= -1

    def set_reflective_BC_y_top(self, U):
        ys, ye, mb = self.get_interior_bounds_y(U)
        U[:, :, ye:] = U[:, :, ye-1:ye-mb-1:-1]
        U[2, :, ye:] *= -1

    def set_continuous_BC_x_bottom(self, U):
        xs, xe, mb = self.get_interior_bounds_x(U)
        U[:, :xs, :] = U[:, xs:xs+1, :]

    def set_continuous_BC_x_top(self, U):
        xs, xe, mb = self.get_interior_bounds_x(U)
        U[:, xe:, :] = U[:, xe-1:xe, :]

    def set_continuous_BC_y_bottom(self, U):
        ys, ye, mb = self.get_interior_bounds_y(U)
        U[:, :, :ys] = U[:, :, ys:ys+1]

    def set_continuous_BC_y_top(self, U):
        ys, ye, mb = self.get_interior_bounds_y(U)
        U[:, :, ye:] = U[:, :, ye-1:ye]

    def get_interior_bounds_x(self, U):
        mb = (U.shape[1] - self.nx)//2
        return mb, mb + self.nx, mb

    def get_interior_bounds_y(self, U):
        mb = (U.shape[2] - self.ny)//2
        return mb, mb + self.ny, mb

    def one_up_x(self, U, pad=0):
        xs, xe, mb = self.get_interior_bounds_x(U)
        return U[:, xs-pad+1:xe+pad+1, :]

    def one_up_y(self, U, pad=0):
        ys, ye, mb = self.get_interior_bounds_y(U)
        return U[:, :, ys-pad+1:ye+pad+1]

    def one_dn_x(self, U, pad=0):
        xs, xe, mb = self.get_interior_bounds_x(U)
        return U[:, xs-pad-1:xe+pad-1, :]

    def one_dn_y(self, U, pad=0):
        ys, ye, mb = self.get_interior_bounds_y(U)
        return U[:, :, ys-pad-1:ye+pad-1]

    def mid_x(self, U, pad=0):
        xs, xe, mb = self.get_interior_bounds_x(U)
        return U[:, xs-pad:xe+pad, :]

    def mid_y(self, U, pad=0):
        ys, ye, mb = self.get_interior_bounds_y(U)
        return U[:, :, ys-pad:ye+pad]

    def interior(self, var):
        xs, xe, mb = self.get_interior_bounds_x(var)
        ys, ye, mb = self.get_interior_bounds_y(var)
        return var[0, xs:xe, ys:ye]

    def diff_up_x(self, U, pad=0):
        return self.one_up_x(U, pad=pad) - self.mid_x(U, pad=pad)

    def diff_up_y(self, U, pad=0):
        return self.one_up_y(U, pad=pad) - self.mid_y(U, pad=pad)

    def diff_dn_x(self, U, pad=0):
        return self.mid_x(U, pad=pad) - self.one_dn_x(U, pad=pad)

    def diff_dn_y(self, U, pad=0):
        return self.mid_y(U, pad=pad) - self.one_dn_y(U, pad=pad)

    def get_timestep(self, U, C):
        c_max_x, c_max_y, c_max_z = self.max_courant_numbers(U)
        return C*min(self.dx/c_max_x, self.dy/c_max_y)


class MHDSystem2D(System2D):

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

    def __init__(self, adiabatic_index=1.4, permeability=1, **grid_kwargs):
        System2D.__init__(self, **grid_kwargs)
        self.adiabatic_index = adiabatic_index
        self.permeability = permeability
        self.U = np.zeros((8, self.nxb, self.nyb)) # [rho, rho*vx, rho*vy, rho*vz, e, Bx, By, Bz]

    def density(self, U):
        return U[0:1, :, :]

    def momentum_x(self, U):
        return U[1:2, :, :]

    def momentum_y(self, U):
        return U[2:3, :, :]

    def momentum_z(self, U):
        return U[3:4, :, :]

    def energy(self, U):
        return U[4:5, :, :]

    def B_field_x(self, U):
        return U[5:6, :, :]

    def B_field_y(self, U):
        return U[6:7, :, :]

    def B_field_z(self, U):
        return U[7:8, :, :]

    def velocity_x(self, U):
        return self.momentum_x(U)/self.density(U)

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

    def compute_flux_x(self, U, F):

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

    def compute_flux_y(self, U, F):

        vx = self.velocity_x(U)
        vy = self.velocity_y(U)
        vz = self.velocity_z(U)
        Bx = self.B_field_x(U)
        By = self.B_field_y(U)
        Bz = self.B_field_z(U)
        p_tot = self.total_pressure(U)

        F[0, :] = self.density(U)*vy
        F[1, :] = self.momentum_z(U)*vy - By*Bx/self.permeability
        F[2, :] = self.momentum_x(U)*vy - By*By/self.permeability + p_tot
        F[3, :] = self.momentum_y(U)*vy - By*Bz/self.permeability
        F[4, :] = (self.energy(U) + p_tot)*vy - By*(By*vy + Bz*vz + Bx*vx)/self.permeability
        F[5, :] = Bx*vy - By*vx
        F[6, :] = 0
        F[7, :] = Bz*vy - By*vz

    def flux_x(self, U):
        F = np.zeros(U.shape)
        self.compute_flux_x(U, F)
        return F

    def flux_y(self, U):
        F = np.zeros(U.shape)
        self.compute_flux_y(U, F)
        return F

    def max_courant_numbers(self, U):
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
        B2_inv = np.zeros(B2.shape)
        B2_inv[B2 > 0] = 1/B2[B2 > 0]
        alfven_speed_squared = B2/(self.permeability*self.density(U))
        sound_speed_squared = self.adiabatic_index*self.thermal_pressure(U)/self.density(U)
        squared_speed_sum = sound_speed_squared + alfven_speed_squared
        fast_mode_wave_speeds_x = np.sqrt((squared_speed_sum + np.sqrt(squared_speed_sum*squared_speed_sum - 4*sound_speed_squared*alfven_speed_squared*Bx2*B2_inv))/2)
        fast_mode_wave_speeds_y = np.sqrt((squared_speed_sum + np.sqrt(squared_speed_sum*squared_speed_sum - 4*sound_speed_squared*alfven_speed_squared*By2*B2_inv))/2)
        fast_mode_wave_speeds_z = np.sqrt((squared_speed_sum + np.sqrt(squared_speed_sum*squared_speed_sum - 4*sound_speed_squared*alfven_speed_squared*Bz2*B2_inv))/2)
        return np.max(flow_speeds_x + fast_mode_wave_speeds_x), \
               np.max(flow_speeds_y + fast_mode_wave_speeds_y), \
               np.max(flow_speeds_z + fast_mode_wave_speeds_z)

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
        return self.interior(vars_funcs[var_name](self.U))

    def init_double_mach_strong_shock_reflection(self, angle=60, x_shock=1/6, rho_pre=1.4, rho_post=8, v_post=8.25, p_pre=1, p_post=116.5):

        xx, yy = np.meshgrid(self.x, self.y, indexing='xy')

        angle_rad = angle*np.pi/180

        vx_pre = 0
        vx_post = v_post*np.sin(angle_rad)
        vy_pre = 0
        vy_post = -v_post*np.cos(angle_rad)
        vz = 0
        Bx = 0
        By = 0
        Bz = 0

        pre_shock = xx < x_shock + yy/np.tan(angle_rad)
        post_shock = np.logical_not(pre_shock)

        #post_shock_lower

        self.U[0, pre_shock] = rho_pre
        self.U[0, post_shock] = rho_post
        self.U[1, pre_shock] = rho_pre*vx_pre
        self.U[1, post_shock] = rho_post*vx_post
        self.U[2, pre_shock] = rho_pre*vy_pre
        self.U[2, post_shock] = rho_post*vy_post
        self.U[3, pre_shock] = rho_pre*vz
        self.U[3, post_shock] = rho_post*vz
        self.U[4, pre_shock] = self.compute_total_energy(rho_pre, p_pre, vx_pre, vy_pre, vz, Bx, By, Bz)
        self.U[4, post_shock] = self.compute_total_energy(rho_post, p_post, vx_post, vy_post, vz, Bx, By, Bz)
        self.U[5, pre_shock] = Bx
        self.U[5, post_shock] = Bx
        self.U[6, pre_shock] = By
        self.U[6, post_shock] = By
        self.U[7, pre_shock] = Bz
        self.U[7, post_shock] = Bz

        def set_BC(self, U):
            self.set_continuous_BC_x_top(U)

        self.init_func = self.init_double_mach_strong_shock_reflection

    def init_square_wave(self, x0=0.35, x1=0.65, y0=0.35, y1=0.65, rho0=1, p0=1, p1=2):

        xx, yy = np.meshgrid(self.x, self.y, indexing='xy')

        region_1 = np.logical_and(np.logical_and(xx > x0, xx < x1), np.logical_and(yy > y0, yy < y1))
        region_0 = np.logical_not(region_1)

        vx = 0
        vy = 0
        vz = 0
        Bx = 0
        By = 0
        Bz = 0

        self.U[0, region_0] = rho0
        self.U[0, region_1] = rho0
        self.U[1, region_0] = rho0*vx
        self.U[1, region_1] = rho0*vx
        self.U[2, region_0] = rho0*vy
        self.U[2, region_1] = rho0*vy
        self.U[3, region_0] = rho0*vz
        self.U[3, region_1] = rho0*vz
        self.U[4, region_0] = self.compute_total_energy(rho0, p0, vx, vy, vz, Bx, By, Bz)
        self.U[4, region_1] = self.compute_total_energy(rho0, p1, vx, vy, vz, Bx, By, Bz)
        self.U[5, region_0] = Bx
        self.U[5, region_1] = Bx
        self.U[6, region_0] = By
        self.U[6, region_1] = By
        self.U[7, region_0] = Bz
        self.U[7, region_1] = Bz

        def set_BC(U):
            self.set_reflective_BC_x_bottom(U)
            self.set_reflective_BC_x_top(U)
            self.set_reflective_BC_y_bottom(U)
            self.set_reflective_BC_y_top(U)

        self.set_BC = set_BC

        self.init_func = self.init_square_wave


class TVDSolver:

    def __init__(self, system, slope_limiter='S'):

        self.s = system
        self.slope_limiter_name = slope_limiter
        self.slope_limiters = {'M': self.slope_limiter_minmod,
                               'U': self.slope_limiter_UMIST,
                               'W': self.slope_limiter_Woodward,
                               'S': self.slope_limiter_superbee}
        self.slope_limiter = self.slope_limiters[slope_limiter]

        self.one_up = [self.s.one_up_x, self.s.one_up_y]
        self.one_dn = [self.s.one_dn_x, self.s.one_dn_y]
        self.mid = [self.s.mid_x, self.s.mid_y]
        self.diff_up = [self.s.diff_up_x, self.s.diff_up_y]
        self.diff_dn = [self.s.diff_dn_x, self.s.diff_dn_y]
        self.flux = [self.s.flux_x, self.s.flux_y]
        self.d = [self.s.dx, self.s.dy]

    def init(self, *args, **kwargs):
        self.t = 0
        self.s.init_func(*args, **kwargs)
        self.s.set_BC(self.s.U)

    def step(self, C):

        dt = self.s.get_timestep(self.s.U, C)

        #self.s.U[:, self.s.xs:self.s.xe, :] += self.MUSCL_full_step(self.s.U, dt/2, 0)
        #self.s.set_BC(self.s.U)

        #self.s.U[:, :, self.s.ys:self.s.ye] += self.MUSCL_full_step(self.s.U, dt,  1)
        #self.s.set_BC(self.s.U)

        #self.s.U[:, self.s.xs:self.s.xe, :] += self.MUSCL_full_step(self.s.U, dt/2,  0)
        #self.s.set_BC(self.s.U)

        self.s.U[:, :, self.s.ys:self.s.ye] += self.MUSCL_full_step(self.s.U, dt/2,  1)
        self.s.set_BC(self.s.U)

        self.s.U[:, self.s.xs:self.s.xe, :] += self.MUSCL_full_step(self.s.U, dt, 0)
        self.s.set_BC(self.s.U)

        self.s.U[:, :, self.s.ys:self.s.ye] += self.MUSCL_full_step(self.s.U, dt/2,  1)
        self.s.set_BC(self.s.U)

        self.t += dt

        return dt

    def MUSCL_predictor_step(self, U, dU_limited, dt, dim):
        mid = self.mid[dim]
        flux = self.flux[dim]
        F_up = flux(mid(U, pad=1) + dU_limited/2)
        F_dn = flux(mid(U, pad=1) - dU_limited/2)
        return -dt/(2*self.d[dim])*(F_up - F_dn)

    def MUSCL_full_step(self, U, dt, dim):

        one_up = self.one_up[dim]
        one_dn = self.one_dn[dim]
        mid = self.mid[dim]
        diff_up = self.diff_up[dim]
        diff_dn = self.diff_dn[dim]
        flux = self.flux[dim]

        dU_up = diff_up(U, pad=1)
        dU_dn = diff_dn(U, pad=1)
        dU_limited_c = self.slope_limiter(dU_up, dU_dn)

        U_halfstep_c = mid(U, pad=1) + self.MUSCL_predictor_step(U, dU_limited_c, dt, dim)

        self.s.set_BC(U_halfstep_c)

        U_L_up = mid(U_halfstep_c) + mid(dU_limited_c)/2
        U_R_up = one_up(U_halfstep_c) - one_up(dU_limited_c)/2
        U_L_dn = one_dn(U_halfstep_c) + one_dn(dU_limited_c)/2
        U_R_dn = mid(U_halfstep_c) - mid(dU_limited_c)/2

        F_LR_up = (flux(U_L_up) + flux(U_R_up))/2
        F_LR_dn = (flux(U_L_dn) + flux(U_R_dn))/2

        dU_transport_c = -dt/self.d[dim]*(F_LR_up - F_LR_dn)

        phi_LR_up = self.dissipative_limiter_LF(U_L_up, U_R_up, dt, dim)
        phi_LR_dn = self.dissipative_limiter_LF(U_L_dn, U_R_dn, dt, dim)

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
        result[0, :] = np.abs(wi[0])
        for i in range(1, n):
            result[i, :] = sgn*wi[i]
        result = np.min(result, axis=0)
        result[result < 0] = 0
        return sgn*result

    def dissipative_limiter_yee(self, U_left, U_right):
        return U_right - U_left

    def dissipative_limiter_LF(self, U_left, U_right, dt, dim):
        U_LR = (U_left + U_right)/2
        return (dt/self.d[dim])*np.max(self.s.max_courant_numbers(U_LR))*self.dissipative_limiter_yee(U_left, U_right)


class Visualizer:

    def __init__(self, solver):
        self.system = solver.s
        self.solver = solver

    def animate(self, var_name, C,
                      vmin=None, vmax=None,
                      title='',
                      T=1, video_time=6, fps=30,
                      savename=False):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title('{} (limiter = {}, C = {:g})'.format(title, self.solver.slope_limiter_name, C))

        plt.tight_layout()

        self.solver.init()

        im = ax.imshow(self.system.get_var(var_name),
                       extent=[self.system.x_start, self.system.x_end, self.system.y_start, self.system.y_end],
                       vmin=vmin,
                       vmax=vmax,
                       origin='lower',
                       aspect='auto',
                       animated=True,
                       interpolation='none',
                       cmap=plt.get_cmap('gray'))

        time_text = ax.text(0.01, 0.99, '', ha='left', va='top', transform=ax.transAxes)

        n_frames = int(np.ceil(video_time*fps))
        dt_out = T/(n_frames - 1)

        def init():
            return im, time_text

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

                im.set_array(self.system.get_var(var_name))
                time_text.set_text('dt = {:g}\nt = {:g}'.format(dt_avg, self.solver.t))

            return im, time_text

        anim = animation.FuncAnimation(fig, update, init_func=init, blit=True, frames=np.arange(n_frames))

        if savename:
            anim.save(savename, writer=animation.FFMpegWriter(fps=fps,
                                                              bitrate=3200,
                                                              extra_args=['-vcodec', 'libx264']))
        else:
            plt.show()


def test_square_wave(C=0.1, slope_limiter='W'):
    system = MHDSystem2D(nx=200, ny=200)
    system.init_square_wave()
    solver = TVDSolver(system, slope_limiter=slope_limiter)
    visualizer = Visualizer(solver)
    visualizer.animate('thermal_pressure', C, T=1, video_time=8,
                       title='Square wave')#,
                       #savename='square_wave_rho_{}_{:g}.mp4'.format(slope_limiter, C))


if __name__ == '__main__':
    test_square_wave()
