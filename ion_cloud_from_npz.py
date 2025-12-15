import manim.opengl
import numpy as np
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from pathlib import Path
from manim import *
from manim import config

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"

config.media_dir = str(PROJECT_ROOT / "media")

# -----------------------------
# Helpers
# -----------------------------
def load_run(path="runs/hybrid_Dplus_test.npz"):
    """Load simulation output from a .npz file."""
    data = np.load(path, allow_pickle=True)

    positions = data["positions"]      # (n_steps, n_particles, 3)

    # Optional time array
    t = data["t"] if "t" in data else None

    # dt is usually present and scalar
    dt = data["dt"].item() if "dt" in data else None

    # Support both "T_total" (new) and "Ttot" (old)
    if "T_total" in data:
        Ttot = data["T_total"]
    elif "Ttot" in data:
        Ttot = data["Ttot"]
    else:
        Ttot = None
    if Ttot is not None and not np.isscalar(Ttot):
        Ttot = np.array(Ttot).item()

    return positions, t, dt, Ttot



def extract_time_window(positions, frac_start, frac_end, max_frames=300):
    """
    Take a fractional time window [frac_start, frac_end] (0..1)
    of the simulation and downsample it to at most max_frames.
    """
    n_steps = positions.shape[0]
    i0 = int(frac_start * (n_steps - 1))
    i1 = int(frac_end   * (n_steps - 1)) + 1
    i0 = max(0, min(i0, n_steps - 1))
    i1 = max(i0 + 1, min(i1, n_steps))

    window = positions[i0:i1]  # (n_win, n_particles, 3)
    n_win = window.shape[0]

    stride = max(1, n_win // max_frames)
    window_ds = window[::stride]
    return window_ds


def choose_scale_from_xy(pos_window, scene_radius=3.0):
    """
    Given positions (n_frames, n_particles, 3), find a scale factor
    so x,y fit roughly inside [-scene_radius, scene_radius].
    """
    xy = pos_window[..., :2]  # (n_frames, n_particles, 2)
    r = np.linalg.norm(xy, axis=2)  # (n_frames, n_particles)
    rmax = np.max(r)
    if rmax == 0 or not np.isfinite(rmax):
        rmax = 1.0
    scale = scene_radius / rmax
    return scale, scene_radius


# =============================
# Scene 1: Early-time collapse
# =============================
class EarlyCollapse2D(Scene):
    """
    Show the violent early-time cloud behavior: everyone falling in,
    shell formation, etc. We crop to the first ~20% of the sim.
    """
    def construct(self):
        positions, t, dt, Ttot = load_run("runs/hybrid_Dplus_test.npz")

        # Use the first 20% of sim time
        pos_win = extract_time_window(positions, frac_start=0.0, frac_end=0.2,
                                      max_frames=300)
        n_frames, n_particles, _ = pos_win.shape

        # Spatial scaling
        scale, scene_radius = choose_scale_from_xy(pos_win, scene_radius=3.0)

        # Axes
        axes = NumberPlane(
            x_range=[-scene_radius, scene_radius, scene_radius / 3],
            y_range=[-scene_radius, scene_radius, scene_radius / 3],
            background_line_style={"stroke_opacity": 0.3},
        )
        self.add(axes)

        # Initial positions
        initial = pos_win[0]
        dots = VGroup()
        for i in range(n_particles):
            x, y = initial[i, 0] * scale, initial[i, 1] * scale
            dot = Dot(point=np.array([x, y, 0.0]),
                      radius=0.03,
                      color=ORANGE,
                      stroke_width=0)
            dots.add(dot)
        self.add(dots)

        # Frame tracker + updater
        frame_tracker = ValueTracker(0)

        def update_dots(group):
            frame = int(np.clip(frame_tracker.get_value(), 0, n_frames - 1))
            pts = pos_win[frame]
            for i, dot in enumerate(group):
                x, y = pts[i, 0] * scale, pts[i, 1] * scale
                dot.move_to(np.array([x, y, 0.0]))
            return group

        dots.add_updater(update_dots)

        # Animate early collapse over a decent chunk of time
        self.play(
            frame_tracker.animate.set_value(n_frames - 1),
            run_time=8.0,
            rate_func=linear
        )

        dots.remove_updater(update_dots)
        self.wait(1)


# =============================
# Scene 2: Late-time orbiters
# =============================
class LateOrbits2D(Scene):
    """
    Focus on the late-time behavior:
    - Crop to last ~40% of sim time
    - Automatically pick a handful of "most active" particles
      (largest radial range) so we don't just stare at a frozen shell.
    """
    def construct(self):
        positions, t, dt, Ttot = load_run("runs/hybrid_Dplus_test.npz")

        # Use the last 40% of the sim
        pos_win = extract_time_window(positions, frac_start=0.6, frac_end=1.0,
                                      max_frames=300)
        n_frames, n_particles, _ = pos_win.shape

        # Compute radial activity per particle in this window
        xy_win = pos_win[..., :2]           # (n_frames, n_particles, 2)
        r = np.linalg.norm(xy_win, axis=2)  # (n_frames, n_particles)
        r_range = r.max(axis=0) - r.min(axis=0)  # (n_particles,)

        # Choose the top-k most "active" particles
        k = min(8, n_particles)  # show up to 8
        idx_sorted = np.argsort(r_range)
        active_indices = idx_sorted[-k:]   # last k are largest range

        # Reduce to just those particles
        pos_focus = pos_win[:, active_indices, :]  # (n_frames, k, 3)
        n_frames_focus, k_particles, _ = pos_focus.shape

        # Spatial scaling from this subset
        scale, scene_radius = choose_scale_from_xy(pos_focus, scene_radius=3.0)

        # Axes
        axes = NumberPlane(
            x_range=[-scene_radius, scene_radius, scene_radius / 3],
            y_range=[-scene_radius, scene_radius, scene_radius / 3],
            background_line_style={"stroke_opacity": 0.3},
        )
        self.add(axes)

        # Dots for the active particles
        initial = pos_focus[0]
        dots = VGroup()
        colors = [YELLOW, RED, GREEN, BLUE, PURPLE, TEAL, GOLD, MAROON]
        for i in range(k_particles):
            x, y = initial[i, 0] * scale, initial[i, 1] * scale
            dot = Dot(
                point=np.array([x, y, 0.0]),
                radius=0.05,
                color=colors[i % len(colors)],
                stroke_width=0,
            )
            dots.add(dot)
        self.add(dots)

        # Optional: show faint background shell (final positions of all particles)
        xy_final = positions[-1, :, :2]
        scale_all, _ = choose_scale_from_xy(positions, scene_radius=3.0)
        shell_points = VGroup()
        for i in range(n_particles):
            x, y = xy_final[i, 0] * scale_all, xy_final[i, 1] * scale_all
            shell_points.add(Dot(point=np.array([x, y, 0.0]),
                                 radius=0.015,
                                 color=GREY,
                                 stroke_width=0,
                                 fill_opacity=0.25))
        self.add(shell_points)

        # Frame tracker + updater for the active ones
        frame_tracker = ValueTracker(0)

        def update_dots(group):
            frame = int(np.clip(frame_tracker.get_value(), 0, n_frames_focus - 1))
            pts = pos_focus[frame]
            for i, dot in enumerate(group):
                x, y = pts[i, 0] * scale, pts[i, 1] * scale
                dot.move_to(np.array([x, y, 0.0]))
            return group

        dots.add_updater(update_dots)

        # Let the orbiters dance
        self.play(
            frame_tracker.animate.set_value(n_frames_focus - 1),
            run_time=8.0,
            rate_func=linear
        )

        dots.remove_updater(update_dots)
        self.wait(1)

# =============================
# Scene 3: 3D Collapse
# =============================

class EarlyCollapse3D(ThreeDScene):
    """
    3D view of the early-time collapse:
    - Uses full x,y,z from the simulation
    - Shows all particles as dots in 3D
    - OpenGL + interactive_embed lets you drag/zoom the camera
    """
    def construct(self):
        # 1) Load data
        positions, t, dt, Ttot = load_run("runs/hybrid_Dplus_test.npz")
        # positions: (n_steps, n_particles, 3)

        # 2) Take early 20% of sim time, with moderate downsampling
        pos_win = extract_time_window(
            positions,
            frac_start=0.0,
            frac_end=0.0085,
            max_frames=400,
        )
        n_frames, n_particles, _ = pos_win.shape

        # 3) Choose a 3D scale so everything fits in a cube [-R, R]^3
        r = np.linalg.norm(pos_win, axis=2)   # (n_frames, n_particles)
        rmax = np.max(r)
        if rmax == 0 or not np.isfinite(rmax):
            rmax = 1.0

        scene_radius = 3.0
        scale = scene_radius / rmax

        # 4) 3D axes
        axes = ThreeDAxes(
            x_range=[-scene_radius, scene_radius, scene_radius / 3],
            y_range=[-scene_radius, scene_radius, scene_radius / 3],
            z_range=[-scene_radius, scene_radius, scene_radius / 3],
        )
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        self.add(axes)

        # --- NEW: chamber sphere (matches R_outer = 0.07 m) ---
        R_outer_phys = 0.07  # meters, same as in your sim config
        chamber_R_scene = R_outer_phys * scale

        chamber_sphere = Sphere(
            center=ORIGIN,
            radius=chamber_R_scene,
            resolution=(24, 24),
            stroke_width=1,
            stroke_opacity=0.4,
            fill_opacity=0.05,
            color=BLUE,
        )
        self.add(chamber_sphere)

        # 5) Create a Dot for each particle at its initial 3D position
        initial = pos_win[0]  # (n_particles, 3)
        dots = VGroup()
        for i in range(n_particles):
            x, y, z = initial[i] * scale
            dot = Dot(
                point=np.array([x, y, z]),
                radius=0.03,
                color=ORANGE,
                stroke_width=0,
            )
            dots.add(dot)
        self.add(dots)
        initial = pos_win[0]  # (n_particles, 3)
        dots = VGroup()
        for i in range(n_particles):
            x, y, z = initial[i] * scale
            dot = Dot(
                point=np.array([x, y, z]),
                radius=0.03,
                color=ORANGE,
                stroke_width=0,
            )
            dots.add(dot)
        self.add(dots)

        # --- NEW: trails for a few tracked particles ---
        tracked_indices = [0, 1, 2]  # pick any 3 particles to highlight
        trails = VGroup()
        colors = [YELLOW, RED, GREEN]
        for j, idx in enumerate(tracked_indices):
            trail = TracedPath(
                lambda idx=idx: dots[idx].get_center(),
                stroke_color=colors[j % len(colors)],
                stroke_width=2,
                dissipating=True,
                stroke_opacity=[1.0, 0.0],  # fade tail
            )
            trails.add(trail)
        self.add(trails)


        # 6) Frame tracker + updater using full 3D coordinates
        frame_tracker = ValueTracker(0)

        def update_dots(group):
            frame = int(np.clip(frame_tracker.get_value(), 0, n_frames - 1))
            pts = pos_win[frame]  # (n_particles, 3)
            for i, dot in enumerate(group):
                x, y, z = pts[i] * scale
                dot.move_to(np.array([x, y, z]))
            return group

        dots.add_updater(update_dots)

        # 7) Kick off the animation once (so they start moving)
        self.play(
            frame_tracker.animate.set_value(n_frames - 1),
            run_time=12.0,
            rate_func=linear,
        )

        # 8) Drop into interactive mode: drag the camera, zoom, etc.
        self.interactive_embed()



class IECSkeleton3D(ThreeDScene):
    """
    IEC fusor animation with 3-ring grid and short ion trails.

    Run from project root:

        manim --renderer=opengl --disable_caching -ql ion_cloud_from_npz.py IECSkeleton3D -p
    """

    NPZ_STEM = "iec_fusor__v0_20kv__rg_2cm__ro_7cm"  # matches default_npz_filename()
    SCALE = 25.0
    MAX_PARTICLES = 20       # number of ions shown
    TIME_SUBSAMPLE = 5      # show every 10th sim step
    RUN_TIME = 8             # seconds of movie
    TRAIL_STEPS = 25         # how many past positions to show in trail

    def construct(self):
        # --- Load data from NPZ ---
        npz_path = RUNS_DIR / f"{self.NPZ_STEM}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find NPZ file: {npz_path}")

        data = np.load(npz_path)
        raw_positions = data["positions"]          # (T_raw, N, 3)
        name = str(data.get("name", self.NPZ_STEM))

        # --- Downsample in time for visualization ---
        positions = raw_positions[::self.TIME_SUBSAMPLE] * self.SCALE
        T_steps, N, _ = positions.shape

        # --- Choose subset of particles ---
        M = min(self.MAX_PARTICLES, N)
        idx = np.linspace(0, N - 1, M, dtype=int)
        paths = positions[:, idx, :]               # (T_steps, M, 3)

        # This ValueTracker is our "time index" along the trajectory
        frame_idx = ValueTracker(0.0)

        # --- Create dots + trails with updaters ---
        dots = VGroup()
        trails = Group()   # <-- IMPORTANT: Group, NOT VGroup

        # nice repeating color palette
        color_cycle = [YELLOW, GREEN, BLUE, ORANGE, PURPLE, TEAL]

        for j in range(M):
            path_j = paths[:, j, :]      # (T_steps, 3) for particle j
            color = color_cycle[j % len(color_cycle)]

            # Particle dot (3D point)
            dot = Dot3D(
                point=path_j[0],
                radius=0.03,
                color=color,
            )

            def dot_updater(mob, path=path_j):
                k = int(np.clip(frame_idx.get_value(), 0, T_steps - 1))
                mob.move_to(path[k])

            dot.add_updater(dot_updater)
            dots.add(dot)

            # Short trail behind each particle
            trail = OpenGLVMobject()
            trail.set_stroke(color=color, width=1.5, opacity=0.8)
            # initialize with a tiny segment so it's not empty
            trail.set_points_as_corners([path_j[0], path_j[0]])

            def trail_updater(mob, path=path_j):
                k = int(np.clip(frame_idx.get_value(), 0, T_steps - 1))
                start = max(0, k - self.TRAIL_STEPS)
                mob.set_points_as_corners(path[start:k+1])

            trail.add_updater(trail_updater)
            trails.add(trail)

        # --- Chamber + 3-ring skeleton grid geometry ---
        R_outer = 0.07 * self.SCALE
        R_grid  = 0.02 * self.SCALE
        wire_r  = 0.002 * self.SCALE   # visual thickness of grid wires

        axes = ThreeDAxes(
            x_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            y_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            z_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            tips=False,
        )

        # Outer spherical chamber
        outer = Sphere(
            radius=R_outer,
            color=BLUE_E,
            stroke_opacity=0.25,
            fill_opacity=0.05,
        )

        # 3 orthogonal rings approximating the IEC grid
        ring_xy = Torus(
            major_radius=R_grid,
            minor_radius=wire_r,
            color=RED,
            opacity=0.7,
        )  # ring in the xy-plane (around z)

        ring_yz = ring_xy.copy().rotate(PI / 2, axis=RIGHT)  # around x-axis
        ring_xz = ring_xy.copy().rotate(PI / 2, axis=UP)     # around y-axis

        title = Text(name, font_size=28).to_corner(UL)

        # --- Camera + initial scene ---
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)


        # Order matters: chamber + grid first, then trails, then dots
        self.add(axes, outer, ring_xy, ring_yz, ring_xz, title, trails, dots)


        # --- Animate frame index across trajectory ---
        self.play(
            frame_idx.animate.set_value(T_steps - 1),
            run_time=self.RUN_TIME,
            rate_func=linear,
        )
        self.wait(1)

class Polywell3D(ThreeDScene):
    """
    Polywell-style animation: linear electrostatic well + cusp B-field.

    Loads positions from the Polywell run saved by chad_core_sim.py:
        runs/polywell__linear_well___cusp_b.npz

    and shows a subset of ions as moving 3D dots with short trails inside
    a spherical chamber.
    """

    # This stem MUST match the slug produced by default_npz_filename(...)
    NPZ_STEM = "polywell__linear_well___cusp_b"

    # Visual tuning parameters
    SCALE = 40.0         # world-units per meter (bigger => more zoomed-in)
    MAX_PARTICLES = 30   # number of ions to show
    TIME_SUBSAMPLE = 3   # use every 5th simulation step for the movie
    RUN_TIME = 8         # seconds of movie playback
    TRAIL_STEPS = 200     # how many past positions to keep in each trail

    def construct(self):
        # --- Load data from NPZ ---
        npz_path = RUNS_DIR / f"{self.NPZ_STEM}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find NPZ file: {npz_path}")

        data = np.load(npz_path)
        raw_positions = data["positions"]          # (T_raw, N, 3)
        name = str(data.get("name", self.NPZ_STEM))

        # --- Downsample in time for visualization ---
        positions = raw_positions[::self.TIME_SUBSAMPLE] * self.SCALE
        T_steps, N, _ = positions.shape

        # --- Choose subset of particles ---
        M = min(self.MAX_PARTICLES, N)
        paths = positions[:, :M, :]    # (T_steps, M, 3)

        # --- Global frame index driving the animation ---
        frame_idx = ValueTracker(0)

        dots = VGroup()
        trails = VGroup()

        # simple repeating color palette
        color_cycle = [YELLOW, GREEN, BLUE, ORANGE, PURPLE, TEAL]

        # --- Create dots + trails with updaters ---
        for j in range(M):
            path_j = paths[:, j, :]      # (T_steps, 3) trajectory for particle j
            color = color_cycle[j % len(color_cycle)]

            # 3D particle dot
            dot = Dot3D(
                point=path_j[0],
                radius=0.03,
                color=color,
            )

            def dot_updater(mob, path=path_j):
                k = int(np.clip(frame_idx.get_value(), 0, T_steps - 1))
                mob.move_to(path[k])
                return mob

            dot.add_updater(dot_updater)
            dots.add(dot)

            # short trailing path
            trail = OpenGLVMobject(color=color, stroke_width=2.0)
            # initialize with a tiny segment so it's not empty
            trail.set_points_as_corners([path_j[0], path_j[0]])

            def trail_updater(mob, path=path_j):
                k = int(np.clip(frame_idx.get_value(), 0, T_steps - 1))
                start = max(0, k - self.TRAIL_STEPS)
                mob.set_points_as_corners(path[start : k + 1])
                return mob

            trail.add_updater(trail_updater)
            trails.add(trail)

        # --- Chamber geometry (outer sphere only for now) ---
        R_outer = 0.07 * self.SCALE   # must match R_outer in the sim config

        axes = ThreeDAxes(
            x_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            y_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            z_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            tips=False,
        )

        outer = Sphere(
            radius=R_outer,
            color=BLUE_E,
            stroke_opacity=0.25,
            fill_opacity=0.05,
        )

        title = Text(name, font_size=28).to_corner(UL)

        # --- Camera + initial scene ---
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)

        # Order matters: chamber first, then trails, then dots
        self.add(axes, outer, title, trails, dots)

        # --- Animate frame index across trajectory ---
        self.play(
            frame_idx.animate.set_value(T_steps - 1),
            run_time=self.RUN_TIME,
            rate_func=linear,
        )
        self.wait(1)

class ChadCore3D(ThreeDScene):
    """
    Chad-core animation: IEC-style outer shell + six coils at ±x, ±y, ±z.

    Loads positions from:
        runs/chad_core__iec_linear_coils__v1.npz
    which is written by chad_core_sim.py.
    """

    NPZ_STEM = "chad_core__iec_linear_coils__v1"

    # Visual tuning
    SCALE = 40.0          # world-units per meter
    MAX_PARTICLES = 30    # ions to show
    TIME_SUBSAMPLE = 3    # use every 3rd sim step
    RUN_TIME = 8          # seconds of animation
    TRAIL_STEPS = 200     # past points in each trail

    def construct(self):
        # -------------------------------
        # Load NPZ produced by sim code
        # -------------------------------
        npz_path = RUNS_DIR / f"{self.NPZ_STEM}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find NPZ file: {npz_path}")

        data = np.load(npz_path)
        raw_positions = data["positions"]              # (T_raw, N, 3)
        name = str(data.get("name", self.NPZ_STEM))

        # Downsample in time and scale to scene units
        positions = raw_positions[::self.TIME_SUBSAMPLE] * self.SCALE
        T_steps, N, _ = positions.shape

        # Pick subset of particles
        M = min(self.MAX_PARTICLES, N)
        paths = positions[:, :M, :]                     # (T_steps, M, 3)

        # Global frame index
        frame_idx = ValueTracker(0)

        dots = VGroup()
        trails = VGroup()

        color_cycle = [YELLOW, GREEN, BLUE, ORANGE, PURPLE, TEAL]

        # ---------------------------------
        # Create moving dots and trails
        # ---------------------------------
        for j in range(M):
            path_j = paths[:, j, :]     # (T_steps, 3)
            color = color_cycle[j % len(color_cycle)]

            dot = Dot3D(
                point=path_j[0],
                radius=0.03,
                color=color,
                stroke_width=0,
            )

            def dot_updater(mob, path=path_j):
                k = int(np.clip(frame_idx.get_value(), 0, T_steps - 1))
                mob.move_to(path[k])
                return mob

            dot.add_updater(dot_updater)
            dots.add(dot)

            trail = OpenGLVMobject()
            trail.set_stroke(color=color, width=1.5, opacity=0.8)

            def trail_updater(mob, path=path_j):
                k = int(np.clip(frame_idx.get_value(), 0, T_steps - 1))
                start = max(0, k - self.TRAIL_STEPS)
                mob.set_points_as_corners(path[start : k + 1])
                return mob

            trail.add_updater(trail_updater)
            trails.add(trail)

        # ---------------------------------
        # Geometry: chamber + SIX coils
        # ---------------------------------
        R_outer = 0.07 * self.SCALE       # matches sim R_outer
        R_coil  = 0.02 * self.SCALE       # coil radius from make_chad_core_config
        wire_r  = 0.0015 * self.SCALE     # visual thickness

        axes = ThreeDAxes(
            x_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            y_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            z_range=(-R_outer, R_outer, 0.02 * self.SCALE),
            tips=False,
        )

        # Outer spherical chamber
        outer = Sphere(
            radius=R_outer,
            color=BLUE_E,
            stroke_opacity=0.25,
            fill_opacity=0.05,
        )

        # Six coils at ±x, ±y, ±z
        coils = VGroup()

        # Base torus: axis along +z, centered at origin, in xy-plane
        base_torus = Torus(
            major_radius=R_coil,
            minor_radius=wire_r,
            color=GREY_B,
            opacity=0.9,
        )

        # Around z-axis at z = ±R_coil
        coil_z_plus = base_torus.copy().shift(OUT * R_coil)
        coil_z_minus = base_torus.copy().shift(IN * R_coil)
        coils.add(coil_z_plus, coil_z_minus)

        # Around x-axis at x = ±R_coil
        coil_x_plus = (
            base_torus.copy()
            .rotate(PI / 2, axis=UP)      # z-axis → x-axis
            .shift(RIGHT * R_coil)
        )
        coil_x_minus = coil_x_plus.copy().shift(LEFT * 2 * R_coil)
        coils.add(coil_x_plus, coil_x_minus)

        # Around y-axis at y = ±R_coil
        coil_y_plus = (
            base_torus.copy()
            .rotate(PI / 2, axis=RIGHT)   # z-axis → y-axis
            .shift(UP * R_coil)
        )
        coil_y_minus = coil_y_plus.copy().shift(DOWN * 2 * R_coil)
        coils.add(coil_y_plus, coil_y_minus)

        title = Text(name, font_size=28).to_corner(UL)

        # ---------------------------------
        # Camera and animation
        # ---------------------------------
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)

        # NOTE: we now add `coils` instead of ring_xy/yz/xz
        self.add(axes, outer, coils, title, trails, dots)

        self.play(
            frame_idx.animate.set_value(T_steps - 1),
            run_time=self.RUN_TIME,
            rate_func=linear,
        )
        self.wait(1)

  
