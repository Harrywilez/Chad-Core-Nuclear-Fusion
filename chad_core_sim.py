# ===== BEGIN REVISED chad_core_sim.py =====
"""
chad_core_sim.py

Overview
--------
Single-species, non-relativistic charged-particle simulator for fusion-inspired
electrostatic traps (fusor / polywell / "Chad-core" style).

Physics:
    Species:   D+ ions (q = +e, m ≈ 2 m_p)
    Equation:  m dv/dt = q (E + v × B)
    Fields:    Analytic, user-specified E(r) [V/m], B(r) [T]
               - Linear electrostatic well: E = -k r, k [V/m^2]
               - Fusor-like radial field:  E_r ∝ 1/r^2 (simple model)
               - Uniform B along z:        B = (0, 0, B0)

Integrator:
    3D Boris pusher (leapfrog-like, energy- and phase-space-friendly for
    Lorentz-force motion).

Boundaries:
    - Outer spherical chamber of radius R_outer [m], absorbing.
    - Optional inner radius (e.g. fusor grid); can be "absorbing" or "none".

Diagnostics:
    - Confinement times (including still-alive ions)
    - Central occupancy statistics (how often ions visit r < r_center)
    - Fraction of ions lost
    - Initial / final kinetic energies (J and eV)
    - Max radius reached and speed relative to c
    - Optional comparison between analytic and numerical oscillation period
      for the linear well (E = -k r).

Workflow:
    1. Run this file (python chad_core_sim.py).
    2. Inspect printed diagnostics and Matplotlib trajectory plots (2D/3D).
    3. Load the saved .npz output from runs/ in ion_cloud_from_npz.py
       (a Manim script) to produce animations.

All quantities are in SI units. The code uses only numpy and matplotlib
to remain "class-friendly" for Programming for Physics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed to enable 3D)
import csv  # at the top of the file with other imports is fine


# =====================================================
# Physical constants (SI)
# =====================================================

e_charge = 1.602176634e-19   # Coulomb
m_p      = 1.67262192369e-27 # kg
m_D      = 2.0 * m_p         # deuteron mass (approx)
eps0     = 8.8541878128e-12  # F/m
c_light  = 2.99792458e8      # m/s


# =====================================================
# Simulation configuration container
# =====================================================

class SimulationConfig:
    """
    Container for all parameters describing a single simulation run.

    Core fields:
        name : str
            Label for this run (used in plots / output file).
        q, m : float
            Charge [C] and mass [kg] for the species (D+ by default).

        N_particles : int
            Number of ions.

        dt      : float
            Time step [s].
        T_total : float
            Total simulation time [s].

        R_outer : float
            Outer spherical chamber radius [m].
        inner_radius : float or None
            Radius of an inner grid / core [m] (if present).
        inner_boundary_type : {"none", "absorbing", "skeleton", "chad_core_coils"}
            - "absorbing"       : ions crossing r <= inner_radius are removed (solid sphere).
            - "skeleton"        : three-ring “IEC grid” skeleton at radius inner_radius.
            - "chad_core_coils" : coil grid modeled as thin rings at radius inner_radius.
            - "none"            : no physical inner object, only a potential minimum.                
        E_func : callable
            E_func(r) -> E(r) [V/m], where r has shape (N, 3) or (3,).
        B_func : callable
            B_func(r) -> B(r) [T], same shapes as E_func.
        rng_seed : int or None
            Seed for numpy's random number generator (reproducibility).

    Notes
    -----
    The simulation is single-species and non-relativistic. It is up to the user
    to choose dt and field strengths so that |v| << c and trajectories remain
    numerically stable. The helper pick_dt_for_linear_well() can be used for
    the harmonic linear-well case E = -k r.
    """

    def __init__(
        self,
        name,
        q=+e_charge,
        m=m_D,
        N_particles=200,
        dt=1e-9,
        T_total=5e-6,
        R_outer=0.1,
        inner_radius=None,
        inner_boundary_type="none",   # "none", "absorbing", "skeleton"
        grid_wire_radius=None,       # used for skeleton 3-ring grid
        init_sigma=0.01,             # initial position spread [m]
        E_func=None,
        B_func=None,
        rng_seed=0,
    ):
        if inner_boundary_type not in ("none", "absorbing", "skeleton", "chad_core_coils"):
            raise ValueError(
                "inner_boundary_type must be 'none', 'absorbing', 'skeleton', or 'chad_core_coils', "
                f"not {inner_boundary_type!r}"
            )



        self.name = str(name)
        self.q = float(q)
        self.m = float(m)
        self.N_particles = int(N_particles)
        self.dt = float(dt)
        self.T_total = float(T_total)
        self.R_outer = float(R_outer)
        self.inner_radius = inner_radius  # can be None
        self.inner_boundary_type = inner_boundary_type
        self.grid_wire_radius = grid_wire_radius
        self.init_sigma = float(init_sigma)
        self.E_func = E_func
        self.B_func = B_func
        self.rng_seed = rng_seed



# =====================================================
# Field definitions (analytic, simple)
# =====================================================

def E_linear_center_well(r, k=1e5):
    """
    Isotropic linear focusing field:
        E(r) = -k r

    Parameters
    ----------
    r : (..., 3) array_like
        Position(s) [m].
    k : float
        Strength of the harmonic well, with units V/m^2.
        If Phi(r) ~ (k/2) |r|^2, then E = -∇Phi = -k r.

    Returns
    -------
    E : (..., 3) ndarray
        Electric field [V/m].

    Notes
    -----
    This field gives simple harmonic motion for a charged particle with
        ω = sqrt(|q k / m|).
    The helper pick_dt_for_linear_well() chooses dt based on ω.
    """
    r = np.asarray(r)
    return -k * r


def E_fusor_like(r, V0=20e3, R_core=0.02):
    """
    Very rough fusor-like radial field.

    Model:
        - Spherical inner grid at radius R_core held at -V0.
        - Outer wall at large radius ~0 potential.
        - Approximate radial field:
              |E_r| ~ V0 * R_core / r^2
          directed inward (toward the negatively biased inner grid).

    Parameters
    ----------
    r : (..., 3) array_like
        Position(s) [m].
    V0 : float
        Voltage magnitude [V]. Inner grid is at -V0 relative to outer wall.
    R_core : float
        Inner grid radius [m].

    Returns
    -------
    E : (..., 3) ndarray
        Electric field [V/m].

    Notes
    -----
    This is deliberately simple and not an exact Laplace solution; it is
    intended only as a toy "fusor-like" radial focusing field.
    """
    r = np.asarray(r)
    R = np.linalg.norm(r, axis=-1, keepdims=True) + 1e-12
    r_hat = r / R
    E_mag = V0 * R_core / (R**2)
    E_vec = -E_mag * r_hat
    return E_vec


def E_iec_concentric_spheres(r, V0=20e3, R_grid=0.02, R_outer=0.07):
    """
    Analytic IEC-style field between concentric spheres.

    Geometry:
        - Inner grid (negative) at radius R_grid held at -V0.
        - Outer chamber at radius R_outer held at 0 V.
        - Region R_grid < r < R_outer solved with Laplace's equation assuming
          spherical symmetry.

    Resulting potential:
        Phi(r) = A + B / r  with boundary conditions:
            Phi(R_outer) = 0,   Phi(R_grid) = -V0

    Radial electric field:
        E_r(r) = -dPhi/dr = -V0 * R_grid * R_outer / (R_outer - R_grid) * 1/r^2
        (pointing inward, toward the negatively biased inner grid).

    In vector form:
        E(r) = -C * r / |r|^3,   where C = V0 * R_grid * R_outer / (R_outer - R_grid).

    Inside the grid (r < R_grid) we approximate Phi ≈ -V0 (conductor),
    so E ≈ 0 in this toy model.
    """
    r = np.asarray(r, dtype=float)
    R = np.linalg.norm(r, axis=-1, keepdims=True) + 1e-20  # avoid divide-by-zero

    C = V0 * R_grid * R_outer / (R_outer - R_grid)
    E_vec = -C * r / (R**3)

    # Zero field inside the grid (treat it as a conductor at -V0).
    mask_inside = (R < R_grid)[..., 0]
    if np.any(mask_inside):
        E_vec[mask_inside] = 0.0

    return E_vec


def E_chad_core(r, V0=30e3, R_core=0.01, R_outer=0.07, k_linear=1e5, w_fusor=0.4):
    """
    Hybrid electric field for the Chad-core configuration.

    We blend:
        - an IEC-like concentric-spheres field (grid at -V0, wall at 0 V), and
        - a central linear harmonic well E = -k_linear * r.

    Specifically:
        E_chad(r) = w_fusor * E_iec_concentric_spheres(r; V0, R_core, R_outer)
                    + (1 - w_fusor) * E_linear_center_well(r; k_linear).

    Near the outer wall and grid, the IEC piece dominates; very close to the
    origin the linear term softens the potential and provides a clean analytic
    time scale via ω = sqrt(|q k_linear / m|).
    """
    r = np.asarray(r, dtype=float)
    E_iec = E_iec_concentric_spheres(r, V0=V0, R_grid=R_core, R_outer=R_outer)
    E_lin = E_linear_center_well(r, k=k_linear)
    return w_fusor * E_iec + (1.0 - w_fusor) * E_lin

    
def B_uniform_z(r, B0=0.5):
    """
    Uniform magnetic field along +z:
        B = (0, 0, B0).

    Parameters
    ----------
    r : (..., 3) array_like
        Position(s) [m]. Only the shape matters here.
    B0 : float
        Magnetic field strength [T].

    Returns
    -------
    B : (..., 3) ndarray
        Magnetic field [T].
    """
    r = np.asarray(r)
    B = np.zeros_like(r)
    B[..., 2] = B0
    return B


def B_three_simple_coils(r, B0=0.5, Rc=0.04, n=4):
    """
    Toy Polywell / cusp-like magnetic field.

    This is a simple analytic stand-in for a six-coil "wiffle-ball" geometry.
    Near the origin the field is approximately linear,

        B_lin(r) = B0 * (x/Rc, y/Rc, -2 z/Rc),

    which is divergence-free. A smooth radial envelope

        f(R) = 1 / (1 + (R/Rc)^n),  R = |r|

    tapers the field off towards the chamber wall.

    Parameters
    ----------
    r : (..., 3) array_like
        Position(s) [m].
    B0 : float
        Nominal field scale [T].
    Rc : float
        Characteristic core / coil radius [m].
    n : float
        Exponent controlling how quickly the field saturates away from
        the center.

    Returns
    -------
    B : (..., 3) ndarray
        Magnetic field [T].
    """
    r = np.asarray(r, dtype=float)
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    # Radius and smooth envelope
    R = np.linalg.norm(r, axis=-1, keepdims=True) + 1e-20
    envelope = 1.0 / (1.0 + (R / Rc) ** n)

    # Linear cusp core, multiplied by envelope
    Bx = envelope[..., 0] * (B0 * x / Rc)
    By = envelope[..., 0] * (B0 * y / Rc)
    Bz = envelope[..., 0] * (B0 * (-2.0 * z / Rc))

    B = np.stack((Bx, By, Bz), axis=-1)
    return B

def B_chad_core_coils_field(r, B0=0.5, R_coil=0.02):
    """
    Approximate magnetic field from six current-carrying coils ("rigatoni")
    arranged on ±x, ±y, ±z using magnetic dipole approximations.

    Geometry:
        Coil centers at:
            (+R_coil, 0, 0), (-R_coil, 0, 0),
            (0, +R_coil, 0), (0, -R_coil, 0),
            (0, 0, +R_coil), (0, 0, -R_coil)

        Each coil is approximated as a dipole with its moment vector m
        aligned with the coil axis. The field from each dipole is added,
        then scaled by B0 as an overall "strength" knob.

    Notes
    -----
    - This is NOT a full Biot–Savart torus solution; it's a fast,
      qualitative model that ties the B-field geometry to the actual
      coil positions used in chad_core_coil_hits.
    - B0 controls the global amplitude of the resulting field.
    """
    r = np.asarray(r, dtype=float)
    B = np.zeros_like(r)

    # Flatten to (N,3) to simplify vectorized math
    r_flat = r.reshape(-1, 3)
    B_flat = np.zeros_like(r_flat)

    # Coil centers (same as rigatoni positions)
    centers = np.array([
        [ R_coil, 0.0    , 0.0    ],
        [-R_coil, 0.0    , 0.0    ],
        [ 0.0   , R_coil , 0.0    ],
        [ 0.0   ,-R_coil , 0.0    ],
        [ 0.0   , 0.0    , R_coil ],
        [ 0.0   , 0.0    ,-R_coil ],
    ], dtype=float)

    # Coil axes (direction of dipole moment)
    axes = np.array([
        [ 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0,-1.0, 0.0],
        [ 0.0, 0.0, 1.0],
        [ 0.0, 0.0,-1.0],
    ], dtype=float)

    # Magnitude of each dipole moment (dimensionless here; B0 is the scale)
    m_mag = 1.0
    m_vecs = m_mag * axes

    # Small cutoff to avoid insane fields exactly at the coil centers
    # (particles shouldn't live *inside* coils anyway)
    eps = 1e-4 * R_coil

    for c, m in zip(centers, m_vecs):
        R = r_flat - c  # (N,3)
        R2 = np.sum(R * R, axis=1, keepdims=True)
        R_mag = np.sqrt(R2)  # (N,1)

        # Avoid division by very small radii
        R_mag = np.maximum(R_mag, eps)
        R_hat = R / R_mag

        # Dipole field pattern (up to an overall constant factor)
        m_dot_Rhat = np.sum(m * R_hat, axis=1, keepdims=True)
        B_i = (3.0 * m_dot_Rhat * R_hat - m) / (R_mag**3)

        B_flat += B_i

    # Global amplitude knob
    B_flat *= B0

    # Reshape back to original shape
    B = B_flat.reshape(r.shape)
    return B


# =====================================================
# Initialization helpers
# =====================================================

def init_positions_gaussian(N, sigma=0.01, rng=None):
    """
    Draw initial positions from an isotropic Gaussian centered at the origin.

    Parameters
    ----------
    N : int
        Number of particles.
    sigma : float
        Standard deviation of each coordinate [m].
    rng : numpy.random.Generator or None
        Random number generator (for reproducibility).

    Returns
    -------
    r0 : (N, 3) ndarray
        Initial positions [m].
    """
    if rng is None:
        rng = np.random.default_rng()
    r0 = rng.normal(0.0, sigma, size=(N, 3))
    return r0


def init_velocities_isotropic(N, v_th=3e4, rng=None):
    """
    Draw initial velocities with isotropic directions and a characteristic speed.

    Parameters
    ----------
    N : int
        Number of particles.
    v_th : float
        "Thermal" speed scale [m/s].
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    v0 : (N, 3) ndarray
        Initial velocities [m/s].

    Notes
    -----
    Speeds are drawn from a Gaussian centered at v_th with a modest spread
    (σ ≈ 0.3 v_th), then oriented isotropically. For the default v_th = 3e4 m/s,
    the typical v/c ≈ 10^-4, safely non-relativistic.
    """
    if rng is None:
        rng = np.random.default_rng()

    phi   = rng.uniform(0.0, 2.0 * np.pi, size=N)
    costh = rng.uniform(-1.0, 1.0, size=N)
    sinth = np.sqrt(1.0 - costh**2)

    speeds = np.abs(rng.normal(loc=v_th, scale=0.3 * v_th, size=N))
    vx = speeds * sinth * np.cos(phi)
    vy = speeds * sinth * np.sin(phi)
    vz = speeds * costh
    return np.stack([vx, vy, vz], axis=1)


def generate_initial_conditions(N, sigma=0.03, v_th=3e4, rng=None):
    """
    Generate a single set of initial conditions (r0, v0) to be reused
    across multiple configurations.

    Parameters
    ----------
    N : int
        Number of particles.
    sigma : float
        Position Gaussian sigma [m].
    v_th : float
        Velocity scale [m/s].
    rng : numpy.random.Generator or None

    Returns
    -------
    r0 : (N, 3) ndarray
    v0 : (N, 3) ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    r0 = init_positions_gaussian(N, sigma=sigma, rng=rng)
    v0 = init_velocities_isotropic(N, v_th=v_th, rng=rng)
    return r0, v0


# =====================================================
# Boris pusher (core integrator)
# =====================================================

def boris_push(r, v, q_over_m, dt, E_func, B_func):
    """
    Perform one Boris step for all particles.

    Parameters
    ----------
    r : (N, 3) ndarray
        Particle positions at time step n.
    v : (N, 3) ndarray
        Particle velocities at time step n.
    q_over_m : float or (N,) array_like
        Charge-to-mass ratio [C/kg].
    dt : float
        Time step [s].
    E_func : callable
        E_func(r) -> E(r) [V/m], evaluated at positions r.
    B_func : callable
        B_func(r) -> B(r) [T], evaluated at positions r.

    Returns
    -------
    r_new : (N, 3) ndarray
        Positions at time step n+1.
    v_new : (N, 3) ndarray
        Velocities at time step n+1.

    Notes
    -----
    The Boris algorithm splits the Lorentz force into:
        - Half electric acceleration
        - Magnetic rotation
        - Half electric acceleration
    which preserves phase-space volume and is widely used in plasma simulations.
    """
    r = np.asarray(r)
    v = np.asarray(v)
    qom = np.asarray(q_over_m)

    N = r.shape[0]
    if qom.ndim == 0:
        qom = np.full(N, qom, dtype=float)

    # 1) Evaluate fields at current positions
    E = E_func(r)
    B = B_func(r)

    # 2) Half electric kick
    qm_col = qom[:, None]  # (N, 1)
    v_minus = v + qm_col * E * (0.5 * dt)

    # 3) Magnetic rotation
    t = qm_col * B * (0.5 * dt)                 # (N, 3)
    t_mag2 = np.sum(t * t, axis=1, keepdims=True)
    s = 2.0 * t / (1.0 + t_mag2)               # (N, 3)

    v_prime = v_minus + np.cross(v_minus, t)
    v_plus  = v_minus + np.cross(v_prime, s)

    # 4) Second half electric kick
    v_new = v_plus + qm_col * E * (0.5 * dt)

    # 5) Position update
    r_new = r + v_new * dt
    return r_new, v_new


# =====================================================
# Period estimation helper (for linear well)
# =====================================================

def estimate_period_from_traj(traj, dt, particle_index=0, axis=0):
    """
    Estimate an oscillation period from a 1D coordinate time series.

    Parameters
    ----------
    traj : (N_steps+1, N_particles, 3) ndarray
        Recorded positions.
    dt : float
        Time step [s].
    particle_index : int
        Which particle to track.
    axis : int
        Which coordinate (0=x, 1=y, 2=z).

    Returns
    -------
    T_num : float
        Estimated period [s], or np.nan if not enough oscillations are visible.

    Method
    ------
    Detect zero-crossings of x(t) and assume a roughly sinusoidal signal.
    Consecutive zero-crossings are separated by T/2, so:
        T ≈ 2 * mean(Δt_between_zero_crossings).
    """
    x = traj[:, particle_index, axis]
    # Ignore exact zeros; look for sign changes
    sgn = np.sign(x)
    # (x_n * x_{n+1} < 0) indicates a sign change
    crossings = np.where(x[:-1] * x[1:] < 0.0)[0]

    if crossings.size < 4:
        return float("nan")

    t_cross = crossings * dt
    dt_cross = np.diff(t_cross)
    T_est = 2.0 * np.mean(dt_cross)
    return float(T_est)




# =====================================================
# Main simulation loop
# =====================================================

def run_simulation(config: SimulationConfig, r0=None, v0=None):
    """
    Run a single non-relativistic D+ ion simulation.

    Parameters
    ----------
    config : SimulationConfig
        All parameters describing the run.
    r0 : (N_particles, 3) array_like or None
        Optional explicit initial positions [m]. If provided, this array is
        copied and used instead of drawing from init_positions_gaussian().
    v0 : (N_particles, 3) array_like or None
        Optional explicit initial velocities [m/s]. If provided, this array is
        copied and used instead of drawing from init_velocities_isotropic().

    Returns
    -------
    traj : (N_steps+1, N_particles, 3) ndarray
    diagnostics : dict
    """
    cfg = config
    if cfg.E_func is None:
        raise ValueError("SimulationConfig.E_func must not be None.")
    if cfg.B_func is None:
        raise ValueError("SimulationConfig.B_func must not be None.")

    rng = np.random.default_rng(cfg.rng_seed)

    # Derived quantities
    q_over_m = cfg.q / cfg.m
    N_steps = int(np.ceil(cfg.T_total / cfg.dt))

    # Initialize particles
    if r0 is None:
        r = init_positions_gaussian(cfg.N_particles, sigma=cfg.init_sigma, rng=rng)
    else:
        r = np.array(r0, dtype=float, copy=True)

    if v0 is None:
        v = init_velocities_isotropic(cfg.N_particles, v_th=3e4, rng=rng)
    else:
        v = np.array(v0, dtype=float, copy=True)



    # --- Initial physical scales (for diagnostics) ---
    r0_mag = np.linalg.norm(r, axis=1)
    v0_mag = np.linalg.norm(v, axis=1)
    initial_r_rms  = float(np.sqrt(np.mean(r0_mag**2)))
    initial_r_max  = float(np.max(r0_mag))
    initial_v_rms  = float(np.sqrt(np.mean(v0_mag**2)))
    initial_v_mean = float(np.mean(v0_mag))

    # Mean initial kinetic energy per ion
    KE0 = 0.5 * cfg.m * v0_mag**2
    initial_mean_ke_J  = float(np.mean(KE0))
    initial_mean_ke_eV = initial_mean_ke_J / e_charge

    # Trajectory history (positions only; velocities are kept only at endpoints)
    traj = np.empty((N_steps + 1, cfg.N_particles, 3), dtype=float)
    traj[0] = r

    # Loss tracking
    alive = np.ones(cfg.N_particles, dtype=bool)
    loss_time = np.full(cfg.N_particles, np.nan, dtype=float)

    # Use local aliases to avoid attribute lookups in the main loop
    E_func = cfg.E_func
    B_func = cfg.B_func
    dt = cfg.dt

    # Global maximum radius tracker (for diagnostics)
    max_radius = float(np.max(r0_mag))

    # Time integration
    for n in range(1, N_steps + 1):
        idx_alive = np.where(alive)[0]
        if idx_alive.size == 0:
            # No particles left; keep positions constant for the remainder
            traj[n] = traj[n - 1]
            continue

        r_alive = r[idx_alive]
        v_alive = v[idx_alive]

        r_new, v_new = boris_push(
            r_alive,
            v_alive,
            q_over_m,
            dt,
            E_func,
            B_func,
        )

        # Update global max radius
        R_mag_new = np.linalg.norm(r_new, axis=1)
        max_radius = max(max_radius, float(np.max(R_mag_new)))

        # Outer wall = absorbing
        hit_outer = R_mag_new >= cfg.R_outer

        # Inner boundary handling
        hit_inner = np.zeros_like(hit_outer, dtype=bool)
        if cfg.inner_radius is not None:
            if cfg.inner_boundary_type == "absorbing":
                # Full spherical absorbing inner core
                hit_inner = R_mag_new <= cfg.inner_radius

            elif cfg.inner_boundary_type == "skeleton":
                # Three-ring IEC skeleton grid (simple ring hits)
                if cfg.grid_wire_radius is None:
                    raise ValueError(
                        "grid_wire_radius must be set when inner_boundary_type='skeleton'."
                    )
                hit_inner = skeleton_three_ring_hits(
                    r_new, cfg.inner_radius, cfg.grid_wire_radius
                )

            elif cfg.inner_boundary_type == "chad_core_coils":
                # Chad-core: thin coil set modeled as six toroidal coils.
                # Particles are only removed if they come within grid_wire_radius
                # of any coil; they can pass between coils and through the center.
                if cfg.grid_wire_radius is None:
                    raise ValueError(
                        "grid_wire_radius must be set when inner_boundary_type='chad_core_coils'."
                    )
                hit_inner = chad_core_coil_hits(
                    r_new, cfg.inner_radius, cfg.grid_wire_radius
                )


            elif cfg.inner_boundary_type == "chad_core_coils":
                # Chad-core: thin coil set, modeled as three orthogonal rings.
                # Particles are only removed if they come within grid_wire_radius
                # of any ring; they can slip between coils and through the center.
                if cfg.grid_wire_radius is None:
                    raise ValueError(
                        "grid_wire_radius must be set when inner_boundary_type='chad_core_coils'."
                    )
                hit_inner = chad_core_coil_hits(
                    r_new, cfg.inner_radius, cfg.grid_wire_radius
                )


        lost = hit_outer | hit_inner


        if np.any(lost):
            lost_global = idx_alive[lost]
            alive[lost_global] = False
            loss_time[lost_global] = n * dt

        survivors = ~lost
        r[idx_alive[survivors]] = r_new[survivors]
        v[idx_alive[survivors]] = v_new[survivors]

        # Record positions (dead particles keep last position they had)
        traj[n] = r

    # =============================
    # Diagnostics
    # =============================

    # Confinement time: lost ions get their loss_time; survivors get T_total
    conf_time = loss_time.copy()
    still_alive = np.isnan(conf_time)
    conf_time[still_alive] = cfg.T_total

    avg_tau    = float(np.mean(conf_time))
    median_tau = float(np.median(conf_time))
    fraction_lost = float(np.count_nonzero(~alive) / cfg.N_particles)

    # Central occupancy:
    # - center_fraction: fraction of all (particle, time) samples with r <= r_center
    # - time_fraction_with_any_in_center: fraction of times at which at least
    #   one ion is inside r <= r_center
    r_center = 0.01  # 1 cm
    N_time = N_steps + 1
    center_counts = 0
    timesteps_with_any_in_center = 0

    for n in range(N_time):
        r_n = traj[n]
        rmag = np.linalg.norm(r_n, axis=1)
        in_center = rmag <= r_center
        center_counts += int(np.sum(in_center))
        if np.any(in_center):
            timesteps_with_any_in_center += 1

    total_samples = cfg.N_particles * N_time
    center_fraction = center_counts / max(total_samples, 1)
    time_fraction_with_any_in_center = timesteps_with_any_in_center / N_time

    #Average time each particle spends in the central sphere r <= r_center
    avg_center_time_s = center_fraction * cfg.T_total

    # --- Density diagnostics ---
    # Volume of central sphere
    V_center = (4.0 / 3.0) * np.pi * (r_center ** 3)

    # Average number of particles in the center at any given time
    avg_particles_in_center = center_fraction * cfg.N_particles

    # Time-averaged central plasma density [1/m^3]
    avg_center_density_m3 = avg_particles_in_center / V_center

    # Global average density if particles were spread throughout the chamber
    V_outer = (4.0 / 3.0) * np.pi * (cfg.R_outer ** 3)
    global_avg_density_m3 = cfg.N_particles / V_outer

    # Final velocity statistics
    v_final_mag = np.linalg.norm(v, axis=1)
    final_v_rms  = float(np.sqrt(np.mean(v_final_mag**2)))
    final_v_mean = float(np.mean(v_final_mag))
    max_speed_over_c = float(np.max(v_final_mag) / c_light)

    # Final kinetic energies
    KEf = 0.5 * cfg.m * v_final_mag**2
    final_mean_ke_J  = float(np.mean(KEf))
    final_mean_ke_eV = final_mean_ke_J / e_charge

    diagnostics = {
        "name": cfg.name,
        "dt_s": cfg.dt,
        "T_total_s": cfg.T_total,
        "N_steps": N_steps,
        "R_outer_m": cfg.R_outer,
        "inner_radius_m": cfg.inner_radius,
        "inner_boundary_type": cfg.inner_boundary_type,
        "avg_confinement_time_s": avg_tau,
        "median_confinement_time_s": median_tau,
        "fraction_lost": fraction_lost,
        "center_radius_m": r_center,
        "center_fraction": center_fraction,
        "time_fraction_with_any_in_center": time_fraction_with_any_in_center,
        "avg_center_time_s": avg_center_time_s,
        "center_volume_m3": V_center,
        "outer_volume_m3": V_outer,
        "avg_particles_in_center": avg_particles_in_center,
        "avg_center_density_m3": avg_center_density_m3,
        "global_avg_density_m3": global_avg_density_m3,
        "max_radius_m": max_radius,
        "initial_r_rms_m": initial_r_rms,
        "initial_r_max_m": initial_r_max,
        "initial_speed_rms_m_per_s": initial_v_rms,
        "initial_speed_mean_m_per_s": initial_v_mean,
        "final_speed_rms_m_per_s": final_v_rms,
        "final_speed_mean_m_per_s": final_v_mean,
        "initial_mean_ke_J": initial_mean_ke_J,
        "initial_mean_ke_eV": initial_mean_ke_eV,
        "final_mean_ke_J": final_mean_ke_J,
        "final_mean_ke_eV": final_mean_ke_eV,
        "max_speed_over_c": max_speed_over_c,
    }

    return traj, diagnostics


# =====================================================
# Plotting helpers
# =====================================================

def plot_xy_projection(traj, cfg: SimulationConfig, max_trajs=20):
    """
    Plot an x-y projection of a subset of trajectories.

    Parameters
    ----------
    traj : (N_steps+1, N_particles, 3) ndarray
        Recorded positions.
    cfg : SimulationConfig
        Configuration (used for R_outer and inner_radius).
    max_trajs : int
        Maximum number of particle trajectories to show.
    """
    N_steps, N, _ = traj.shape
    n_show = min(max_trajs, N)

    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n_show):
        xs = traj[:, i, 0]
        ys = traj[:, i, 1]
        ax.plot(xs, ys, lw=0.7, alpha=0.8)

    # Outer boundary circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(
        cfg.R_outer * np.cos(theta),
        cfg.R_outer * np.sin(theta),
        "k--",
        alpha=0.5,
        label="Outer wall",
    )

    # Inner boundary circle (if present)
    if cfg.inner_radius is not None:
        ax.plot(
            cfg.inner_radius * np.cos(theta),
            cfg.inner_radius * np.sin(theta),
            "r--",
            alpha=0.5,
            label="Inner core",
        )

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"XY projection: {cfg.name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_3d_trajectories(traj, cfg: SimulationConfig, max_trajs=20,
                         elev=25, azim=45):
    """
    3D trajectory plot for a subset of particles.

    Parameters
    ----------
    traj : (N_steps+1, N_particles, 3) ndarray
        Recorded positions.
    cfg : SimulationConfig
        Configuration (used for chamber radii).
    max_trajs : int
        Maximum number of trajectories to plot.
    elev : float
        Elevation angle for the 3D view.
    azim : float
        Azimuthal angle for the 3D view.
    """
    N_steps, N, _ = traj.shape
    n_show = min(max_trajs, N)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectories
    for i in range(n_show):
        xs = traj[:, i, 0]
        ys = traj[:, i, 1]
        zs = traj[:, i, 2]
        ax.plot(xs, ys, zs, lw=0.7, alpha=0.8)

    # Parameter grid for spheres
    u = np.linspace(0, 2 * np.pi, 40)
    v_ang = np.linspace(0, np.pi, 20)

    # Outer spherical chamber (wireframe)
    Ro = cfg.R_outer
    xo = Ro * np.outer(np.cos(u), np.sin(v_ang))
    yo = Ro * np.outer(np.sin(u), np.sin(v_ang))
    zo = Ro * np.outer(np.ones_like(u), np.cos(v_ang))
    ax.plot_wireframe(xo, yo, zo, color="k", alpha=0.25, linewidth=0.5)

    # Optional inner core / grid (wireframe)
    if cfg.inner_radius is not None:
        Ri = cfg.inner_radius
        xi = Ri * np.outer(np.cos(u), np.sin(v_ang))
        yi = Ri * np.outer(np.sin(u), np.sin(v_ang))
        zi = Ri * np.outer(np.ones_like(u), np.cos(v_ang))
        ax.plot_wireframe(xi, yi, zi, color="r", alpha=0.4, linewidth=0.7)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"3D trajectories: {cfg.name}")

    # Lock axes to full chamber so you see the whole shell
    R = cfg.R_outer
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_box_aspect([1, 1, 1])

    # Camera angle
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


# =====================================================
# Time-step helper for seperate cases
# =====================================================

def pick_dt_for_linear_well(q, m, k, steps_per_period=300):
    """
    Choose a time step for the linear well E = -k r.

    For a charged particle of charge q and mass m in the field E = -k r, the
    motion is harmonic with angular frequency:
        ω = sqrt(|q k / m|).

    Parameters
    ----------
    q : float
        Charge [C].
    m : float
        Mass [kg].
    k : float
        Strength of the linear well [V/m^2].
    steps_per_period : int
        Desired number of time steps per oscillation period.

    Returns
    -------
    dt : float
        Time step [s].
    T : float
        Oscillation period [s].
    """
    omega = np.sqrt(abs(q * k / m))
    T = 2.0 * np.pi / omega
    dt = T / steps_per_period
    return dt, T

def pick_dt_for_polywell(
    q,
    m,
    k,
    B0,
    steps_per_linear_period=300,
    steps_per_gyro=300,
):
    """
    Choose a time step for a configuration with both a linear electrostatic
    well E = -k r and a magnetic field of scale B0.

    We combine two characteristic timescales:

      - Linear-well oscillation:
            ω_E = sqrt(|q k / m|),
            T_E = 2π / ω_E.

      - Cyclotron motion in the magnetic field:
            ω_c = |q| B0 / m,
            T_c = 2π / ω_c.

    The time step is chosen as

        dt = min(T_E / steps_per_linear_period,
                 T_c / steps_per_gyro).

    Parameters
    ----------
    q : float
        Charge [C].
    m : float
        Mass [kg].
    k : float
        Linear-well strength [V/m^2].
    B0 : float
        Magnetic field scale [T].
    steps_per_linear_period : int
        Target number of steps per linear-well period.
    steps_per_gyro : int
        Target number of steps per cyclotron period.

    Returns
    -------
    dt : float
        Recommended time step [s].
    T_E : float
        Linear-well period [s].
    T_c : float
        Cyclotron period [s] (np.inf if B0 == 0).
    """
    # Linear-well timescale
    dt_E, T_E = pick_dt_for_linear_well(
        q, m, k, steps_per_period=steps_per_linear_period
    )

    # Magnetic timescale
    if B0 == 0.0:
        # No magnetic field: fall back to linear-well choice.
        return dt_E, T_E, np.inf

    omega_c = abs(q * B0 / m)
    T_c = 2.0 * np.pi / omega_c
    dt_c = T_c / steps_per_gyro

    dt = min(dt_E, dt_c)
    return dt, T_E, T_c


# =====================================================
# 3-ring Skeleton Grid collision helper
# =====================================================

def skeleton_three_ring_hits(r, R_grid, a_wire):
    """
    Determine which particles hit a three-ring skeleton grid.

    Geometry:
        - Three circular rings of radius R_grid:
          * Ring in the xy-plane (around the z-axis)
          * Ring in the yz-plane (around the x-axis)
          * Ring in the zx-plane (around the y-axis)
        - Each ring has an effective wire radius a_wire.

    Parameters
    ----------
    r : (N, 3) array_like
        Particle positions [m].
    R_grid : float
        Grid radius [m].
    a_wire : float
        Effective wire radius [m]. Particles within distance <= a_wire
        of any ring are considered to collide with the grid.

    Returns
    -------
    hits : (N,) ndarray of bool
        True where the particle is counted as hitting the grid.
    """
    r = np.asarray(r, dtype=float)
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    # Distances to each ring (squared).
    # 1) Ring in xy-plane, centered on origin, around z-axis.
    rho_xy = np.sqrt(x * x + y * y)
    d_xy2 = (rho_xy - R_grid) ** 2 + z * z

    # 2) Ring in yz-plane, around x-axis.
    rho_yz = np.sqrt(y * y + z * z)
    d_yz2 = (rho_yz - R_grid) ** 2 + x * x

    # 3) Ring in zx-plane, around y-axis.
    rho_zx = np.sqrt(z * z + x * x)
    d_zx2 = (rho_zx - R_grid) ** 2 + y * y

    d_min2 = np.minimum(np.minimum(d_xy2, d_yz2), d_zx2)
    return d_min2 <= a_wire ** 2


def chad_core_coil_hits(r, R_coil, a_wire, length_factor=0.6):
    """
    Determine which particle positions lie inside any of six finite-length
    cylindrical coils ("rigatoni") arranged on ±x, ±y, ±z.

    Geometry (approximate):
        - Coil centers at:
            (+R_coil, 0, 0), (-R_coil, 0, 0),
            (0, +R_coil, 0), (0, -R_coil, 0),
            (0, 0, +R_coil), (0, 0, -R_coil)

        - Each coil is a straight cylinder of radius a_wire and length
          L = length_factor * R_coil, whose axis is along the coordinate axis
          that passes through its center.

        - A particle is considered to hit a coil if it lies within radius
          a_wire of the coil axis AND within ±L/2 along that axis.

    Parameters
    ----------
    r : (N, 3) array_like
        Particle positions [m].
    R_coil : float
        Distance of coil centers from the origin [m].
    a_wire : float
        Effective coil radius [m] (cylinder radius).
    length_factor : float
        Sets the coil length L = length_factor * R_coil. Must be < 2
        if you want coils not to overlap the origin.

    Returns
    -------
    hits : (N,) ndarray of bool
        True where a particle lies inside at least one coil volume.
    """
    r = np.asarray(r, dtype=float)
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    L = length_factor * R_coil
    half_L = 0.5 * L

    hits = np.zeros(x.shape, dtype=bool)

    # --- Coils on ±x, axis along x ---
    # Center at (+R_coil, 0, 0)
    d_x = x - R_coil
    radial_sq = y**2 + z**2
    coil_pos_x = (np.abs(d_x) <= half_L) & (radial_sq <= a_wire**2)

    # Center at (-R_coil, 0, 0)
    d_x = x + R_coil
    radial_sq = y**2 + z**2
    coil_neg_x = (np.abs(d_x) <= half_L) & (radial_sq <= a_wire**2)

    hits |= coil_pos_x | coil_neg_x

    # --- Coils on ±y, axis along y ---
    # Center at (0, +R_coil, 0)
    d_y = y - R_coil
    radial_sq = x**2 + z**2
    coil_pos_y = (np.abs(d_y) <= half_L) & (radial_sq <= a_wire**2)

    # Center at (0, -R_coil, 0)
    d_y = y + R_coil
    radial_sq = x**2 + z**2
    coil_neg_y = (np.abs(d_y) <= half_L) & (radial_sq <= a_wire**2)

    hits |= coil_pos_y | coil_neg_y

    # --- Coils on ±z, axis along z ---
    # Center at (0, 0, +R_coil)
    d_z = z - R_coil
    radial_sq = x**2 + y**2
    coil_pos_z = (np.abs(d_z) <= half_L) & (radial_sq <= a_wire**2)

    # Center at (0, 0, -R_coil)
    d_z = z + R_coil
    radial_sq = x**2 + y**2
    coil_neg_z = (np.abs(d_z) <= half_L) & (radial_sq <= a_wire**2)

    hits |= coil_pos_z | coil_neg_z

    return hits


    
# =====================================================
# Config suite over initial conditions helper
# =====================================================

def run_config_suite_over_initial_conditions(
    config_builders,
    n_realizations=10,
    N_particles=200,
    init_sigma=0.03,
    v_th=3e4,
    base_seed=0,
    metrics=("avg_confinement_time_s", "fraction_lost"),
    out_csv=None,
):
    """
    Run multiple configurations over many shared initial conditions.

    For each realization k:
        1) Generate a single (r0, v0) set with given N_particles, sigma, v_th.
        2) For each config label in config_builders:
            - build a fresh SimulationConfig via config_builders[label]()
            - run_simulation(cfg, r0, v0)
            - record selected metrics.

    Parameters
    ----------
    config_builders : dict[str, callable]
        Mapping from a label (e.g. "linear", "iec_skeleton") to a zero-argument
        function that returns a SimulationConfig.
    n_realizations : int
        Number of different initial condition sets (A, B, C, ...).
    N_particles : int
        Number of particles (must match cfg.N_particles for all configs).
    init_sigma : float
        Shared initial position sigma [m] used for all realizations.
    v_th : float
        Shared initial velocity scale [m/s].
    base_seed : int
        Base seed; each realization uses base_seed + k.
    metrics : tuple[str]
        Diagnostics keys to record.
    out_csv : str or None
        If not None, path to a CSV file to write rows:
            case_label, realization, metric_name, value

    Returns
    -------
    results : dict[str, dict[str, list[float]]]
        results[case_label][metric_name] is a list over realizations.
    """
    # Prepare results structure
    results = {
        label: {m: [] for m in metrics}
        for label in config_builders.keys()
    }

    # Optional CSV writer
    csv_writer = None
    csv_file = None
    if out_csv is not None:
        directory = os.path.dirname(out_csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        csv_file = open(out_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["case", "realization", "metric", "value"])

    try:
        for k in range(n_realizations):
            rng = np.random.default_rng(base_seed + k)
            r0, v0 = generate_initial_conditions(
                N_particles, sigma=init_sigma, v_th=v_th, rng=rng
            )

            for label, builder in config_builders.items():
                cfg = builder()
                if cfg.N_particles != N_particles:
                    raise ValueError(
                        f"Config {label!r} has N_particles={cfg.N_particles}, "
                        f"but N_particles={N_particles} was requested."
                    )

                traj, diagnostics = run_simulation(cfg, r0=r0, v0=v0)

                for m in metrics:
                    val = diagnostics.get(m, np.nan)
                    results[label][m].append(float(val))
                    if csv_writer is not None:
                        csv_writer.writerow([label, k, m, val])
    finally:
        if csv_file is not None:
            csv_file.close()

    return results


# =====================================================
# Saving for Manim
# =====================================================

def save_trajectory_npz(filename, traj, cfg: SimulationConfig, diagnostics):
    """
    Save trajectory and basic metadata to a compressed .npz file.

    Parameters
    ----------
    filename : str
        Path to output .npz file (e.g. "runs/chad_linear_well.npz").
    traj : (N_steps+1, N_particles, 3) ndarray
        Particle positions.
    cfg : SimulationConfig
        Configuration object.
    diagnostics : dict
        Diagnostics dictionary (not fully saved, but could be expanded).

    Notes
    -----
    The Manim script ion_cloud_from_npz.py expects at least:
        - positions: (N_steps+1, N_particles, 3)
        - dt: time step [s]
        - T_total: total simulation time [s]
        - name: run label
    """
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    np.savez_compressed(
        filename,
        positions=traj,
        dt=cfg.dt,
        T_total=cfg.T_total,
        name=np.array(cfg.name),
        # Note: diagnostics are not saved yet to keep the file lightweight.
        # They could be added here as additional fields if desired.
    )


def default_npz_filename(cfg: SimulationConfig, runs_dir="runs", tag=None):
    """
    Construct a default .npz filename from a SimulationConfig name.

    Parameters
    ----------
    cfg : SimulationConfig
        The configuration whose name will be used.
    runs_dir : str
        Directory to place the file in (default "runs").
    tag : str or None
        Optional extra tag to append (e.g. "v1", "scanA").

    Returns
    -------
    filename : str
        Something like "runs/linear_well_no_inner_core.npz".
    """
    # Simple slugify: keep alphanumerics, turn everything else into "_"
    raw = cfg.name.lower()
    slug_chars = []
    for ch in raw:
        if ch.isalnum():
            slug_chars.append(ch)
        else:
            slug_chars.append("_")
    slug = "".join(slug_chars).strip("_")
    if tag is not None:
        slug = f"{slug}_{tag}"
    return os.path.join(runs_dir, f"{slug}.npz")


# =====================================================
# Diagnostics printing helper
# =====================================================

def print_diagnostics(diagnostics, T_analytic=None, T_numeric=None):
    """
    Pretty-print diagnostics for quick inspection and presentations.

    Parameters
    ----------
    diagnostics : dict
        Diagnostics dictionary produced by run_simulation().
    T_analytic : float or None
        Analytic period [s] (for linear well), if known.
    T_numeric : float or None
        Numerically estimated period [s], if computed.
    """
    name = diagnostics.get("name", "<unnamed>")
    print("\n=== Numerical diagnostics for run ===")
    print(f"Run name: {name}")
    print("--------------------------------------")

    # Choose a friendly print order for key scalars
    keys_order = [
        "dt_s",
        "T_total_s",
        "N_steps",
        "R_outer_m",
        "inner_radius_m",
        "inner_boundary_type",
        "avg_confinement_time_s",
        "median_confinement_time_s",
        "fraction_lost",
        "center_radius_m",
        "center_fraction",
        "time_fraction_with_any_in_center",
        "max_radius_m",
        "initial_r_rms_m",
        "initial_r_max_m",
        "initial_speed_rms_m_per_s",
        "initial_speed_mean_m_per_s",
        "final_speed_rms_m_per_s",
        "final_speed_mean_m_per_s",
        "initial_mean_ke_J",
        "initial_mean_ke_eV",
        "final_mean_ke_J",
        "final_mean_ke_eV",
        "max_speed_over_c",
    ]

    for key in keys_order:
        if key not in diagnostics:
            continue
        val = diagnostics[key]
        if isinstance(val, float):
            # Use fixed or scientific notation depending on magnitude
            if abs(val) < 1e-3 or abs(val) > 1e3:
                s_val = f"{val: .3e}"
            else:
                s_val = f"{val: .6f}"
        else:
            s_val = str(val)
        print(f"{key:35s}: {s_val}")

    # Print any extra keys that were not in the default order
    extra_keys = [k for k in diagnostics.keys() if k not in keys_order + ["name"]]
    if extra_keys:
        print("\nAdditional diagnostics:")
        for k in sorted(extra_keys):
            print(f"  {k}: {diagnostics[k]}")

    # Optional period comparison
    if T_analytic is not None or T_numeric is not None:
        print("\nPeriod comparison (linear well):")
        if T_analytic is not None:
            print(f"  Analytic period T_analytic [s] : {T_analytic: .6e}")
        if T_numeric is not None and np.isfinite(T_numeric):
            print(f"  Numeric period  T_numeric [s]  : {T_numeric: .6e}")
        if T_analytic is not None and T_numeric is not None and np.isfinite(T_numeric):
            rel_err = abs(T_numeric - T_analytic) / T_analytic
            print(f"  Relative error |ΔT|/T          : {rel_err: .3e}")
        elif T_numeric is not None and not np.isfinite(T_numeric):
            print("  Numeric period estimate not reliable (insufficient crossings).")


# =====================================================
# Convenience wrapper: run + report
# =====================================================

def run_and_report(cfg: SimulationConfig, k_linear=None, steps_per_period=None):
    """
    Run a simulation, estimate period (for linear well), and print diagnostics.

    Parameters
    ----------
    cfg : SimulationConfig
        Configuration for the run.
    k_linear : float or None
        If not None, treat the field as E = -k_linear r and compute the
        analytic period for comparison.
    steps_per_period : float or None
        Optional reference target for steps per period; if provided, it is
        displayed alongside the actual T/dt.

    Returns
    -------
    traj : (N_steps+1, N_particles, 3) ndarray
        Particle positions over time.
    diagnostics : dict
        Diagnostics dictionary.
    """
    print("==============================================")
    print(f"Starting simulation: {cfg.name}")
    print("==============================================")

    T_analytic = None

    if k_linear is not None:
        # Analytic scales for linear well
        omega = np.sqrt(abs(cfg.q * k_linear / cfg.m))
        T_analytic = 2.0 * np.pi / omega
        steps_per = T_analytic / cfg.dt

        print("\n=== Analytic linear-well scales (deuteron) ===")
        print(f"k [V/m^2]       = {k_linear: .3e}")
        print(f"q/m [C/kg]      = {(cfg.q / cfg.m): .3e}")
        print(f"omega [rad/s]   = {omega: .3e}")
        print(f"Period T [s]    = {T_analytic: .3e}")
        print(f"Chosen dt [s]   = {cfg.dt: .3e}")
        print(f"steps/period    = {steps_per: .1f}")
        if steps_per_period is not None:
            print(f"(target steps/period ≈ {steps_per_period})")
        print(f"Total time [s]  = {cfg.T_total: .3e}")
        print(f"N_steps         = {int(np.ceil(cfg.T_total/cfg.dt))}")

    # Run the actual simulation
    traj, diagnostics = run_simulation(cfg)

    # Estimate period from the trajectory (using x-coordinate of particle 0)
    T_numeric = None
    if k_linear is not None:
        T_numeric = estimate_period_from_traj(traj, cfg.dt, particle_index=0, axis=0)

    # Print diagnostics in a neat block
    print_diagnostics(diagnostics, T_analytic=T_analytic, T_numeric=T_numeric)

    return traj, diagnostics


# =====================================================
# Standard configuration factories
# =====================================================

def make_linear_well_config(
    name="Linear well + no inner core",
    k=1e5,
    steps_per_period=400,
    N_periods=2.0,
    N_particles=200,
    R_outer=0.07,
    B0=0.0,
    rng_seed=42,
):
    """
    Factory for the baseline linear-well configuration used as a benchmark.

    Parameters
    ----------
    name : str
        Run label.
    k : float
        Linear-well strength [V/m^2] in E = -k r.
    steps_per_period : int
        Target number of timesteps per oscillation.
    N_periods : float
        Total number of oscillation periods to simulate.
    N_particles : int
        Number of ions.
    R_outer : float
        Outer chamber radius [m].
    B0 : float
        Uniform Bz field strength [T] (set to 0.0 for pure electrostatic case).
    rng_seed : int or None
        Random seed for reproducible initialization.

    Returns
    -------
    cfg : SimulationConfig
        Ready-to-run configuration.
    T : float
        Analytic oscillation period [s].
    """
    dt, T = pick_dt_for_linear_well(e_charge, m_D, k, steps_per_period=steps_per_period)
    cfg = SimulationConfig(
        name=name,
        q=+e_charge,
        m=m_D,
        N_particles=N_particles,
        dt=dt,
        T_total=N_periods * T,
        R_outer=R_outer,
        inner_radius=None,
        inner_boundary_type="none",
        E_func=lambda r: E_linear_center_well(r, k=k),
        B_func=lambda r: B_uniform_z(r, B0=B0),
        rng_seed=rng_seed,
    )
    return cfg, T

def make_polywell_config(
    name="Polywell: linear well + cusp B",
    k=1e5,
    B0=0.5,
    Rc=0.04,
    steps_per_linear_period=400,
    steps_per_gyro=400,
    N_periods=3.0,
    N_particles=200,
    R_outer=0.07,
    rng_seed=7,
):
    """
    Factory for a toy Polywell configuration.

    Field model:
        - Electrostatic: linear center well E = -k r.
        - Magnetic: cusp-like field B_three_simple_coils(...)
          approximating a six-coil Polywell geometry.
        - No physical inner grid (inner_boundary_type='none').

    The time step dt is chosen to resolve both the linear-well oscillation
    period and the cyclotron period at the field scale B0.

    Parameters
    ----------
    name : str
        Label for this run.
    k : float
        Linear-well strength [V/m^2].
    B0 : float
        Magnetic field scale [T].
    Rc : float
        Characteristic coil/core radius [m] used in B_three_simple_coils.
    steps_per_linear_period : int
        Target steps per electrostatic oscillation period.
    steps_per_gyro : int
        Target steps per cyclotron period.
    N_periods : float
        Number of characteristic periods (max(T_E, T_c)) to simulate.
    N_particles : int
        Number of ions.
    R_outer : float
        Outer spherical chamber radius [m].
    rng_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    cfg : SimulationConfig
        Ready-to-run Polywell configuration.
    T_E : float
        Linear-well period [s].
    T_c : float
        Cyclotron period [s].
    """
    q = +e_charge
    m = m_D

    dt, T_E, T_c = pick_dt_for_polywell(
        q=q,
        m=m,
        k=k,
        B0=B0,
        steps_per_linear_period=steps_per_linear_period,
        steps_per_gyro=steps_per_gyro,
    )

    # Simulate a few characteristic periods (take the slower of T_E, T_c)
    T_char = max(T_E, T_c if np.isfinite(T_c) else 0.0)
    T_total = N_periods * T_char

    cfg = SimulationConfig(
        name=name,
        q=q,
        m=m,
        N_particles=N_particles,
        dt=dt,
        T_total=T_total,
        R_outer=R_outer,
        inner_radius=None,
        inner_boundary_type="none",
        E_func=lambda r: E_linear_center_well(r, k=k),
        B_func=lambda r, B0=B0, Rc=Rc: B_three_simple_coils(r, B0=B0, Rc=Rc),
        rng_seed=rng_seed,
    )

    return cfg, T_E, T_c



def make_iec_config(
    name="IEC fusor: V0=20kV, Rg=2cm, Ro=7cm",
    V0=20e3,
    R_grid=0.02,
    R_outer=0.07,
    N_particles=200,
    dt=1e-10,
    T_total=2e-6,
    init_sigma=0.03,
    grid_wire_radius=0.002,  # 2 mm wire thickness as an example
    rng_seed=123,
):
    """
    Factory for an IEC-like configuration using concentric spherical electrodes
    and a three-ring skeleton grid at the inner radius.

    Field model:
        - Inner spherical grid at radius R_grid held at -V0.
        - Outer spherical chamber at radius R_outer held at 0 V.
        - E(r) given by the concentric-spheres Laplace solution:
              E(r) = -C * r / |r|^3  for R_grid < r < R_outer,
          with C = V0 * R_grid * R_outer / (R_outer - R_grid).
        - Inside the grid (r < R_grid), E ≈ 0 in this toy model.

    Grid collisions:
        - Inner boundary uses "skeleton" mode:
          three orthogonal rings of radius R_grid with wire radius grid_wire_radius.
        - Particles are only removed if they pass within a_wire of any ring;
          they can slip through the gaps and continue inward.

    Parameters
    ----------
    name : str
        Run label.
    V0 : float
        Grid bias magnitude [V] (grid at -V0, wall at 0 V).
    R_grid : float
        Inner grid radius [m].
    R_outer : float
        Outer chamber radius [m].
    N_particles : int
        Number of ions.
    dt : float
        Time step [s].
    T_total : float
        Total simulated time [s].
    init_sigma : float
        Initial position spread [m] (Gaussian sigma).
    grid_wire_radius : float
        Effective wire radius of each ring [m].
    rng_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    cfg : SimulationConfig
        Ready-to-run IEC configuration.
    """
    cfg = SimulationConfig(
        name=name,
        q=+e_charge,
        m=m_D,
        N_particles=N_particles,
        dt=dt,
        T_total=T_total,
        R_outer=R_outer,
        inner_radius=R_grid,
        inner_boundary_type="skeleton",
        grid_wire_radius=grid_wire_radius,
        init_sigma=init_sigma,
        E_func=lambda r: E_iec_concentric_spheres(
            r, V0=V0, R_grid=R_grid, R_outer=R_outer
        ),
        B_func=lambda r: B_uniform_z(r, B0=0.0),  # IEC: electrostatic only for now
        rng_seed=rng_seed,
    )
    return cfg

def make_chad_core_config(
    name="Chad-core: IEC + linear well + coils",
    V0=100e3,
    R_coil=0.02,
    R_outer=0.07,
    k_linear=1e5,
    w_fusor=0.4,
    steps_per_period=400,
    N_periods=20.0,
    N_particles=200,
    init_sigma=0.01,
    grid_wire_radius=0.0005,  # thinner "wire" than IEC grid, e.g. 1.5 mm
    B0=0.25,
    rng_seed=321,
):
    """
    Factory for a Chad-core hybrid configuration.

    Geometry:
        - Outer spherical chamber of radius R_outer.
        - Coil/grid radius R_coil: three orthogonal rings of radius R_coil
          with effective wire radius grid_wire_radius.
        - Inner boundary type 'chad_core_coils' removes only particles that
          come within grid_wire_radius of any ring.

    Fields:
        - Electric: E_chad_core(r; V0, R_coil, R_outer, k_linear, w_fusor).
        - Magnetic: B_three_simple_coils(r, B0=B0) (simple cusp-like placeholder).

    Time step:
        - dt is chosen from the linear-well curvature k_linear using
          pick_dt_for_linear_well, which guarantees a reasonable number of
          steps per oscillation in the central harmonic region.
    """
    # Choose dt and period from the linear-well curvature
    dt, T = pick_dt_for_linear_well(
        e_charge,
        m_D,
        k_linear,
        steps_per_period=steps_per_period,
    )

    cfg = SimulationConfig(
        name=name,
        q=+e_charge,
        m=m_D,
        N_particles=N_particles,
        dt=dt,
        T_total=N_periods * T,
        R_outer=R_outer,
        inner_radius=R_coil,
        inner_boundary_type="chad_core_coils",
        grid_wire_radius=grid_wire_radius,
        init_sigma=init_sigma,
        E_func=lambda r: E_chad_core(
            r,
            V0=V0,
            R_core=R_coil,
            R_outer=R_outer,
            k_linear=k_linear,
            w_fusor=w_fusor,
        ),
        B_func=lambda r: B_chad_core_coils_field(r, B0=B0, R_coil=R_coil),
        rng_seed=rng_seed,
    )
    return cfg, T




# =====================================================
# Example usage (can be run as a script)
# =====================================================

if __name__ == "__main__":
    # Example: linear focusing well, no inner object (polywell-like / Chad-core toy model).
    #
    # The choice of k controls the oscillation frequency through:
    #   ω = sqrt(|q k / m|)
    # For D+ with the defaults below, the resulting speeds remain safely
    # non-relativistic for dt chosen by pick_dt_for_linear_well().

    k = 1e5  # [V/m^2]
    steps_per_period_target = 400
    N_periods = 2.0

    cfg, T = make_linear_well_config(
        name="Linear well + no inner core",
        k=k,
        steps_per_period=steps_per_period_target,
        N_periods=N_periods,
        N_particles=200,
        R_outer=0.07,
        B0=0.0,
        rng_seed=42,
    )

    # Run, compare analytic vs numerical period, and print diagnostics
    traj, diag = run_and_report(
        cfg,
        k_linear=k,
        steps_per_period=steps_per_period_target,
    )

    # 2D view (x-y projection)
    plot_xy_projection(traj, cfg, max_trajs=25)

    # 3D view (rotate with mouse in interactive backends)
    plot_3d_trajectories(traj, cfg, max_trajs=25, elev=30, azim=45)

    # Save for Manim animation, using a filename derived from the run name
    out_file = default_npz_filename(cfg)
    save_trajectory_npz(out_file, traj, cfg, diag)
    print(f"\nSaved trajectory to: {out_file}")

if __name__ == "__main__":
    # IEC run (uses fusor-like field with skeleton inner grid)
    cfg = make_iec_config()
    traj, diag = run_and_report(cfg)

    out_file = default_npz_filename(cfg)  # -> runs/iec_fusor__v0_20kv__rg_2cm__ro_7cm.npz
    save_trajectory_npz(out_file, traj, cfg, diag)
    print(f"\nSaved trajectory to: {out_file}")

    # ---------------------------------------------------------
    # Polywell run: linear electrostatic well + cusp-like B
    # ---------------------------------------------------------
    k_poly = 1e5         # [V/m^2]  same order as linear example
    B0_poly = 0.25        # [T]      toy Polywell field scale
    Rc_poly = 0.04       # [m]      "coil radius" inside Ro = 0.07 m
    steps_lin_poly = 400 # steps per linear period
    steps_gyro_poly = 400
    N_periods_poly = 30.0

    cfg_poly, T_E_poly, T_c_poly = make_polywell_config(
        name="Polywell: linear well + cusp B",
        k=k_poly,
        B0=B0_poly,
        Rc=Rc_poly,
        steps_per_linear_period=steps_lin_poly,
        steps_per_gyro=steps_gyro_poly,
        N_periods=N_periods_poly,
        N_particles=200,
        R_outer=0.07,
        rng_seed=99,
    )

    print("\n=== Analytic scales for Polywell run ===")
    print(f"k [V/m^2]           = {k_poly: .3e}")
    print(f"B0 [T]              = {B0_poly: .3e}")
    print(f"q/m [C/kg]          = {(+e_charge/m_D): .3e}")
    print(f"Linear period  T_E  = {T_E_poly: .3e} s")
    if np.isfinite(T_c_poly):
        print(f"Gyro period    T_c  = {T_c_poly: .3e} s")
    else:
        print("Gyro period    T_c  = inf (B0=0)")
    print(f"Chosen dt [s]       = {cfg_poly.dt: .3e}")
    print(f"T_total [s]         = {cfg_poly.T_total: .3e}")
    print(f"N_steps             = {int(np.ceil(cfg_poly.T_total / cfg_poly.dt))}")

    # Run Polywell simulation and report diagnostics
    traj_poly, diag_poly = run_and_report(
        cfg_poly,
        k_linear=k_poly,
        steps_per_period=steps_lin_poly,
    )

    # Quick-look plots for Polywell run
    plot_xy_projection(traj_poly, cfg_poly, max_trajs=20)
    plot_3d_trajectories(traj_poly, cfg_poly, max_trajs=20, elev=25, azim=45)

    out_file_poly = default_npz_filename(cfg_poly)
    save_trajectory_npz(out_file_poly, traj_poly, cfg_poly, diag_poly)
    print(f"\nSaved trajectory to: {out_file_poly}")

    
    out_file_poly = default_npz_filename(cfg_poly)
    save_trajectory_npz(out_file_poly, traj_poly, cfg_poly, diag_poly)
    print(f"\nSaved trajectory to: {out_file_poly}")

    # ---------------------------------------------------------
    # Chad-core run: IEC + linear well + coils
    # ---------------------------------------------------------
    cfg_chad, T_chad = make_chad_core_config(
    name="Chad-core: IEC+linear+coils, v1",
    V0=7e3,
    R_coil=0.02,
    R_outer=0.07,
    k_linear=1e5,
    w_fusor=0.25,
    steps_per_period=400,
    N_periods=2.0,
    N_particles=200,
    init_sigma=0.01,
    grid_wire_radius=0.0005,
    B0=0.0001,
    rng_seed=99,
)


    traj_chad, diag_chad = run_and_report(cfg_chad, k_linear=None)

    # >>> ADD THESE LINES <<<
    # Quick-look plots for Chad-core
    plot_xy_projection(traj_chad, cfg_chad, max_trajs=20)
    plot_3d_trajectories(traj_chad, cfg_chad, max_trajs=20, elev=25, azim=45)
    # <<< END ADDED LINES >>>

    out_file_chad = default_npz_filename(cfg_chad)
    save_trajectory_npz(out_file_chad, traj_chad, cfg_chad, diag_chad)
    print(f"\nSaved Chad-core trajectory to: {out_file_chad}")


# ===== END REVISED chad_core_sim.py =====

