import numpy as np
import matplotlib.pyplot as plt
import string
import os

PLOT_DIR = "runs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# 0) Suite controls
# -----------------------------
N_SUITE_PARTICLES = 200
SIGMA_SUITE = 0.01      # [m] ~1 cm cloud
VTH_SUITE = 3e4         # [m/s] thermal-ish speed
R_SUITE_OUTER = 0.11    # [m] common outer radius for all configs

# how many "letters" (seeds) you want: A, B, C, ...
LETTERS = string.ascii_uppercase[:10]   # A..J, change 10 -> 20 or 26 if you want
N_REALIZATIONS = len(LETTERS)
BASE_SEED = 0  # A → 0, B → 1, etc.  (kth realization uses base_seed + k)

# Top-10 metrics to record from diagnostics
SUITE_METRICS = (
    "avg_confinement_time_s",
    "median_confinement_time_s",
    "fraction_lost",
    "center_fraction",
    "time_fraction_with_any_in_center",
    "avg_center_density_m3",
    "global_avg_density_m3",
    "final_mean_ke_eV",
    "max_radius_m",
    "avg_center_time_s",
)

# -----------------------------
# 1) Builders for each case
#    (make sure these actually use the globals above)
# -----------------------------

def build_linear():
    k_linear = 1e5
    steps_target = 400
    dt_lin, T_lin = pick_dt_for_linear_well(
        e_charge, m_D, k_linear, steps_per_period=steps_target
    )
    N_periods = 2.0
    return SimulationConfig(
        name="Linear well",
        q=e_charge,
        m=m_D,
        N_particles=N_SUITE_PARTICLES,
        dt=dt_lin,
        T_total=N_periods * T_lin,
        R_outer=R_SUITE_OUTER,
        inner_radius=None,
        inner_boundary_type="none",
        E_func=lambda r: E_linear_center_well(r, k=k_linear),
        B_func=lambda r: B_uniform_z(r, B0=0.0),
    )

def build_iec():
    cfg = make_iec_config(
        name="IEC fusor: V0=20kV, Rg=2cm, Ro=11cm",
        V0=20e3,
        R_grid=0.02,
        R_outer=R_SUITE_OUTER,
        N_particles=N_SUITE_PARTICLES,
        dt=1e-10,
        T_total=2e-6,
        init_sigma=SIGMA_SUITE,
        grid_wire_radius=0.002,
        rng_seed=123,
    )
    return cfg

def build_polywell():
    ret = make_polywell_config(
        name="Polywell: linear well + cusp B",
        k=1e5,
        B0=0.5,
        Rc=0.04,
        steps_per_linear_period=400,
        steps_per_gyro=400,
        N_periods=3.0,
        N_particles=N_SUITE_PARTICLES,
        R_outer=R_SUITE_OUTER,
        rng_seed=7,
    )
    return ret if isinstance(ret, SimulationConfig) else ret[0]

def build_chad_core():
    ret = make_chad_core_config(
        name="Chad-core: IEC + linear well + coils",
        V0=100e3,
        R_coil=0.02,
        R_outer=R_SUITE_OUTER,
        k_linear=1e5,
        w_fusor=0.25,
        steps_per_period=400,
        N_periods=20.0,
        N_particles=N_SUITE_PARTICLES,
        init_sigma=SIGMA_SUITE,
        grid_wire_radius=1e-4,
        rng_seed=42,
    )
    return ret if isinstance(ret, SimulationConfig) else ret[0]

config_builders = {
    "linear": build_linear,
    "iec_skeleton": build_iec,
    "polywell": build_polywell,
    "chad_core": build_chad_core,
}

# -----------------------------
# 2) Run suite
# -----------------------------

results = run_config_suite_over_initial_conditions(
    config_builders=config_builders,
    n_realizations=N_REALIZATIONS,
    N_particles=N_SUITE_PARTICLES,
    init_sigma=SIGMA_SUITE,
    v_th=VTH_SUITE,
    base_seed=BASE_SEED,
    metrics=SUITE_METRICS,
    out_csv="runs/case_comparison_full.csv",
)

labels = list(config_builders.keys())
print("Recorded metrics per case:")
for case in labels:
    print(f"  {case}:")
    for m in SUITE_METRICS:
        vals = results[case][m]
        print(f"    {m}: {len(vals)} realizations, "
              f"mean={np.mean(vals):.3e}, std={np.std(vals):.3e}")

# -----------------------------
# 3) Sorted bar charts: mean value per config for each metric
# -----------------------------

for metric in SUITE_METRICS:
    means = np.array([np.mean(results[label][metric]) for label in labels])
    order = np.argsort(means)
    sorted_labels = [labels[i] for i in order]
    sorted_means = means[order]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(sorted_labels))
    plt.bar(x, sorted_means)
    plt.xticks(x, sorted_labels, rotation=20)
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Mean {metric} over {N_REALIZATIONS} seeds (A..{LETTERS[-1]})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric}_means.png"), dpi=300)
    plt.show()
    plt.close()


# -----------------------------
# 4) Optional: histograms over seeds for each metric
# -----------------------------
# This will spam a lot of plots; comment out if too much.

for metric in SUITE_METRICS:
    plt.figure(figsize=(6, 4))
    for label in labels:
        vals = results[label][metric]
        plt.hist(vals, bins=10, alpha=0.5, label=label)
    plt.xlabel(metric.replace("_", " "))
    plt.ylabel("Count over seeds")
    plt.title(f"Distribution of {metric} across seeds and configs")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric}_hist.png"), dpi=300)
    plt.show()
    plt.close()

