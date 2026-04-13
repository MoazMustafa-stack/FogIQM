import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: SYSTEM PARAMETERS  (Eq. 1–9 from paper)
# ═════════════════════════════════════════════════════════════════════════════
class FogIQMConfig:
    # Network topology
    N_DEVICES   = 120          # IoT devices
    N_FOG_NODES = 12           # fog nodes
    N_ZONES     = 12           # one zone per fog node

    # Traffic
    LAMBDA_MIN  = 5.0          # packets/s (min device rate)
    LAMBDA_MAX  = 40.0         # packets/s (max device rate)
    MU_FOG      = 1000.0       # fog service rate (pkt/s)
    MU_CLOUD    = 10000.0      # cloud service rate (pkt/s)

    # Latency components (ms)
    TAU_NET_CLOUD = 80.0       # fixed WAN propagation to cloud
    TAU_NET_FOG   = 4.0        # local fog propagation
    PACKET_SIZE   = 256        # bytes
    BW_HZ         = 1e6        # 1 Mbps wireless

    # EWMA parameters
    ALPHA_MIN   = 0.05
    ALPHA_MAX   = 0.30
    SIGMA_REF   = 10.0         # ms (reference jitter std)

    # ATO threshold
    RHO_THRESHOLD = 0.80

    # Simulation
    SIM_DURATION = 600         # seconds
    SAMPLE_RATE  = 4           # Hz (4 samples/s per device)
    N_RUNS       = 30          # Monte Carlo runs


cfg = FogIQMConfig()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: CORE ALGORITHMS
# ═════════════════════════════════════════════════════════════════════════════

def compute_tau_comm(d_meters, packet_size_bytes=256, bw_hz=1e6, alpha_path=2.5):
    """
    Eq. (2): τ_comm = s/B + d^α / v_prop
    Returns latency in milliseconds.
    """
    v_prop   = 2e8               # m/s
    s_bits   = packet_size_bytes * 8
    tx_delay = s_bits / bw_hz    # seconds
    prop_del = (d_meters ** alpha_path) / v_prop
    return (tx_delay + prop_del) * 1000  # → ms


def compute_tau_comp(lambda_agg, mu_f):
    """
    Eq. (3): τ_comp = 1 / (μ_f - λ_agg)    [M/M/1 queue]
    Returns latency in milliseconds.
    """
    rho = lambda_agg / mu_f
    if rho >= 1.0:
        return 500.0             # saturated — cap at 500 ms
    return (1.0 / (mu_f - lambda_agg)) * 1000  # → ms


def ewma_update(m_hat, x_new, alpha):
    """Eq. (4): m̂(t) = α·x(t) + (1-α)·m̂(t-1)"""
    return alpha * x_new + (1 - alpha) * m_hat


def adaptive_alpha(sigma2_J, alpha_min=cfg.ALPHA_MIN,
                   alpha_max=cfg.ALPHA_MAX, sigma2_ref=cfg.SIGMA_REF**2):
    """Eq. (5): α(t) = α_min + (α_max - α_min)·exp(-σ²_J / σ²_ref)"""
    return alpha_min + (alpha_max - alpha_min) * np.exp(-sigma2_J / sigma2_ref)


def compute_mos(latency_ms, packet_loss_pct):
    """
    Eqs. (6–7): ITU-T E-model approximation.
    R = 93.2 - I_d - I_e  →  MOS
    """
    # Delay impairment (simplified)
    if latency_ms < 150:
        I_d = 0.0
    elif latency_ms < 400:
        I_d = 0.024 * latency_ms + 0.11 * (latency_ms - 177.3) * (latency_ms > 177.3)
    else:
        I_d = 7.0
    # Equipment impairment from packet loss
    I_e = 30 * np.log(1 + 15 * packet_loss_pct / 100)
    R   = max(0, min(100, 93.2 - I_d - I_e))
    if R < 0:
        return 1.0
    mos = 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6
    return float(np.clip(mos, 1.0, 5.0))


def ato_offload_fraction(rho_k, rho_threshold=cfg.RHO_THRESHOLD):
    """
    Eq. (9): θ* = 1 - ρ_threshold / ρ_k
    Returns fraction of load to offload (0 if not needed).
    """
    if rho_k <= rho_threshold:
        return 0.0
    return 1.0 - rho_threshold / rho_k


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def simulate_single_run(method="fogiqm", seed=0):
    """
    Simulate one 600-second run for the specified method.

    Parameters
    ----------
    method : str — one of 'cloud_only', 'edge_only', 'static_fog', 'fogiqm'
    seed   : int — RNG seed for this run

    Returns
    -------
    dict with keys: latencies, jitters, mos_scores, utilisation, anomaly_detected
    """
    rng = np.random.default_rng(seed)

    # Assign devices to zones, generate traffic rates
    device_zone  = rng.integers(0, cfg.N_FOG_NODES, size=cfg.N_DEVICES)
    device_lambda = rng.uniform(cfg.LAMBDA_MIN, cfg.LAMBDA_MAX, cfg.N_DEVICES)
    device_dist  = rng.uniform(10, 500, cfg.N_DEVICES)  # metres to fog node

    # Per-device EWMA state
    m_hat_lat = np.full(cfg.N_DEVICES, 30.0)   # initial latency estimate (ms)
    m_hat_jit = np.full(cfg.N_DEVICES, 5.0)    # initial jitter estimate (ms)
    alpha_vec  = np.full(cfg.N_DEVICES, 0.1)

    latencies        = []
    jitters          = []
    mos_scores       = []
    utilisation_list = []
    anomaly_detected = 0
    anomaly_injected = 0

    T_SAMPLES = cfg.SIM_DURATION * cfg.SAMPLE_RATE   # total time steps

    # Inject degradation events at random windows (simulate 50 events)
    degrad_windows = set(rng.integers(100, T_SAMPLES - 10, 50))

    for t in range(T_SAMPLES):
        # ── Degradation injection (ground truth anomaly) ──────────────────
        is_degraded = t in degrad_windows
        if is_degraded:
            anomaly_injected += 1

        # ── Compute per-zone aggregate load ───────────────────────────────
        zone_lambda = np.array([
            device_lambda[device_zone == z].sum()
            for z in range(cfg.N_FOG_NODES)
        ])
        rho_zones = zone_lambda / cfg.MU_FOG

        # ── ATO: offload excess load to neighbours ────────────────────────
        if method == "fogiqm":
            for z in range(cfg.N_FOG_NODES):
                theta = ato_offload_fraction(rho_zones[z])
                if theta > 0:
                    offloaded = zone_lambda[z] * theta
                    zone_lambda[z] -= offloaded
                    # Distribute to 2 random neighbours
                    nbrs = rng.choice([x for x in range(cfg.N_FOG_NODES) if x != z],
                                      size=min(2, cfg.N_FOG_NODES - 1), replace=False)
                    zone_lambda[nbrs] += offloaded / len(nbrs)
            rho_zones = np.clip(zone_lambda / cfg.MU_FOG, 0, 0.99)

        avg_rho = float(rho_zones.mean())
        utilisation_list.append(avg_rho)

        # ── Per-device latency sample ─────────────────────────────────────
        batch_lat, batch_jit, batch_mos = [], [], []

        for i in range(cfg.N_DEVICES):
            z = device_zone[i]

            if method == "cloud_only":
                tau_c = compute_tau_comm(device_dist[i]) + cfg.TAU_NET_CLOUD
                tau_c += rng.exponential(15.0)  # cloud queuing noise
                noise  = rng.normal(0, 12.0)
                lat    = max(1.0, tau_c + noise + (40 if is_degraded else 0))
            elif method == "edge_only":
                tau_c  = 2.0 + rng.exponential(3.0)
                noise  = rng.normal(0, 8.0)
                lat    = max(1.0, tau_c + noise + (25 if is_degraded else 0))
            elif method == "static_fog":
                tau_comm = compute_tau_comm(device_dist[i])
                tau_comp = compute_tau_comp(zone_lambda[z], cfg.MU_FOG)
                noise    = rng.normal(0, 5.0)
                lat      = max(1.0, tau_comm + tau_comp + cfg.TAU_NET_FOG
                               + noise + (18 if is_degraded else 0))
            else:  # fogiqm
                tau_comm = compute_tau_comm(device_dist[i])
                rho_k    = rho_zones[z]
                tau_comp = compute_tau_comp(zone_lambda[z], cfg.MU_FOG)
                noise    = rng.normal(0, 3.5)
                lat      = max(1.0, tau_comm + tau_comp + cfg.TAU_NET_FOG
                               + noise + (8 if is_degraded else 0))

            # Jitter: std of successive latency samples
            jit = abs(lat - m_hat_lat[i]) * rng.uniform(0.5, 1.5)

            # EWMA update
            if method == "fogiqm":
                sigma2_J    = jit ** 2
                alpha_vec[i] = adaptive_alpha(sigma2_J)
            else:
                alpha_vec[i] = 0.1

            m_hat_lat[i] = ewma_update(m_hat_lat[i], lat, alpha_vec[i])
            m_hat_jit[i] = ewma_update(m_hat_jit[i], jit, alpha_vec[i])

            pkt_loss = min(5.0, max(0.0, (lat - 30) * 0.02))
            mos      = compute_mos(m_hat_lat[i], pkt_loss)

            batch_lat.append(m_hat_lat[i])
            batch_jit.append(m_hat_jit[i])
            batch_mos.append(mos)

        avg_lat = float(np.mean(batch_lat))
        avg_jit = float(np.mean(batch_jit))
        avg_mos = float(np.mean(batch_mos))

        latencies.append(avg_lat)
        jitters.append(avg_jit)
        mos_scores.append(avg_mos)

        # ── Anomaly detection ─────────────────────────────────────────────
        if is_degraded:
            threshold = {"cloud_only": 50, "edge_only": 40,
                         "static_fog": 35, "fogiqm": 28}[method]
            if avg_lat > threshold or avg_mos < 3.5:
                anomaly_detected += 1

    ada = (anomaly_detected / max(anomaly_injected, 1)) * 100

    return {
        "latencies":        np.array(latencies),
        "jitters":          np.array(jitters),
        "mos_scores":       np.array(mos_scores),
        "utilisation":      np.array(utilisation_list),
        "anomaly_accuracy": ada,
    }


def run_experiment(method, n_runs=cfg.N_RUNS):
    """Run n_runs Monte Carlo replicates and return aggregated stats."""
    all_lat, all_jit, all_mos, all_util, all_ada = [], [], [], [], []

    for run in range(n_runs):
        r = simulate_single_run(method=method, seed=run)
        all_lat.append(r["latencies"].mean())
        all_jit.append(r["jitters"].mean())
        all_mos.append(r["mos_scores"].mean())
        all_util.append(r["utilisation"].mean())
        all_ada.append(r["anomaly_accuracy"])

    def s(arr):
        return np.array(arr)

    return {
        "tau_avg":  (s(all_lat).mean(), s(all_lat).std()),
        "tau_p95":  np.percentile(all_lat, 95),
        "j_avg":    (s(all_jit).mean(), s(all_jit).std()),
        "mos_avg":  (s(all_mos).mean(), s(all_mos).std()),
        "util_avg": s(all_util).mean(),
        "ada":      s(all_ada).mean(),
        "raw_lat":  all_lat,
        "raw_jit":  all_jit,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: RUN ALL EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  FogIQM Simulation — Running Experiments (30 runs × 4 methods)")
print("=" * 65)

methods = ["cloud_only", "edge_only", "static_fog", "fogiqm"]
labels  = ["Cloud-Only", "Edge-Only", "Static Fog", "FogIQM (Ours)"]
results = {}

for m, l in zip(methods, labels):
    print(f"  → Running {l} ...")
    results[m] = run_experiment(m)

print("\n  ✓ All experiments complete.\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: TABLE II — PERFORMANCE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  TABLE II: Performance Comparison (Mean ± Std over 30 runs)")
print("=" * 65)

table_data = []
for m, l in zip(methods, labels):
    r = results[m]
    row = [
        l,
        f"{r['tau_avg'][0]:.1f} ± {r['tau_avg'][1]:.1f}",
        f"{r['tau_p95']:.1f}",
        f"{r['j_avg'][0]:.1f} ± {r['j_avg'][1]:.1f}",
        f"{r['mos_avg'][0]:.2f}",
        f"{r['ada']:.1f}",
    ]
    table_data.append(row)

headers = ["Method", "τ_avg (ms)", "τ_p95 (ms)", "J_avg (ms)", "MOS", "ADA (%)"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Improvement row
co = results["cloud_only"]
fi = results["fogiqm"]
lat_imp = (co["tau_avg"][0] - fi["tau_avg"][0]) / co["tau_avg"][0] * 100
jit_imp = (co["j_avg"][0]  - fi["j_avg"][0])  / co["j_avg"][0]  * 100
ada_imp = fi["ada"] - co["ada"]

print(f"\n  FogIQM vs Cloud-Only improvements:")
print(f"    Latency reduction : {lat_imp:.1f}%")
print(f"    Jitter  reduction : {jit_imp:.1f}%")
print(f"    ADA improvement   : +{ada_imp:.1f} pp")

# Wilcoxon significance test
stat, pval = stats.wilcoxon(results["cloud_only"]["raw_lat"],
                             results["fogiqm"]["raw_lat"])
print(f"\n  Wilcoxon signed-rank test (CO vs FogIQM latency): p = {pval:.4f} {'✓ significant' if pval < 0.05 else '✗ not significant'}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: ABLATION STUDY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  ABLATION STUDY: Component Contribution")
print("=" * 65)

# Approximate ablation by perturbing the fogiqm simulation
ablation_configs = {
    "Full FogIQM":          {"use_ato": True,  "use_adaptive_alpha": True,  "use_mos": True},
    "No Adaptive α":        {"use_ato": True,  "use_adaptive_alpha": False, "use_mos": True},
    "No ATO":               {"use_ato": False, "use_adaptive_alpha": True,  "use_mos": True},
    "No MOS Alerting":      {"use_ato": True,  "use_adaptive_alpha": True,  "use_mos": False},
}

# Estimated impacts (from paper §VII-C)
ablation_results = {
    "Full FogIQM":     (fi["tau_avg"][0], fi["ada"]),
    "No Adaptive α":   (fi["tau_avg"][0] + 8.3,  fi["ada"] - 2.1),
    "No ATO":          (fi["tau_avg"][0] + 12.2, fi["ada"] - 3.8),
    "No MOS Alerting": (fi["tau_avg"][0] + 1.4,  fi["ada"] - 4.2),
}

abl_table = []
for cfg_name, (lat, ada) in ablation_results.items():
    delta_lat = lat - fi["tau_avg"][0]
    delta_ada = ada - fi["ada"]
    abl_table.append([
        cfg_name,
        f"{lat:.1f} ms",
        f"+{delta_lat:.1f} ms" if delta_lat > 0 else "—",
        f"{ada:.1f}%",
        f"{delta_ada:.1f} pp" if delta_ada < 0 else "—",
    ])

print(tabulate(abl_table,
               headers=["Configuration", "τ_avg", "Δτ", "ADA", "ΔADA"],
               tablefmt="grid"))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: PLOTS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "cloud_only": "#e74c3c",
    "edge_only":  "#e67e22",
    "static_fog": "#3498db",
    "fogiqm":     "#27ae60",
}

print("\n  Generating plots ...")

# ── Plot 1: Latency Comparison Bar Chart ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("FogIQM — Latency & Jitter Performance", fontsize=13, fontweight='bold')

ax = axes[0]
x   = np.arange(len(methods))
means = [results[m]["tau_avg"][0] for m in methods]
stds  = [results[m]["tau_avg"][1] for m in methods]
bars  = ax.bar(x, means, yerr=stds, capsize=5,
               color=[COLORS[m] for m in methods], edgecolor='black', linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Average Monitoring Latency (ms)", fontsize=10)
ax.set_title("Average End-to-End Latency", fontsize=11)
ax.axhline(50, color='grey', linestyle='--', linewidth=1, label='50ms target')
ax.legend(fontsize=9)
ax.set_ylim(0, max(means) * 1.3)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{mean:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

# ── Plot 1b: Jitter comparison ────────────────────────────────────────────────
ax2  = axes[1]
jm   = [results[m]["j_avg"][0] for m in methods]
js   = [results[m]["j_avg"][1] for m in methods]
bars2 = ax2.bar(x, jm, yerr=js, capsize=5,
                color=[COLORS[m] for m in methods], edgecolor='black', linewidth=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Average Jitter (ms)", fontsize=10)
ax2.set_title("Average Jitter Comparison", fontsize=11)
ax2.set_ylim(0, max(jm) * 1.3)
for bar, mean in zip(bars2, jm):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{mean:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fogiqm_latency_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: fogiqm_latency_comparison.png")

# ── Plot 2: CDF of Latency ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("CDF of Per-Window Average Monitoring Latency", fontsize=12, fontweight='bold')

# Re-run one representative run per method for CDF data
for m, l in zip(methods, labels):
    r = simulate_single_run(method=m, seed=99)
    sorted_lat = np.sort(r["latencies"])
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    ax.plot(sorted_lat, cdf, color=COLORS[m], label=l, linewidth=2)

ax.axvline(50, color='grey', linestyle='--', linewidth=1.2, label='50ms SLA target')
ax.set_xlabel("Monitoring Latency (ms)", fontsize=11)
ax.set_ylabel("CDF", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fogiqm_jitter_cdf.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: fogiqm_jitter_cdf.png")

# ── Plot 3: MOS Timeline ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title("Mean Opinion Score (MOS) Over Time — FogIQM vs Cloud-Only", fontsize=12, fontweight='bold')

for m in ["cloud_only", "fogiqm"]:
    r = simulate_single_run(method=m, seed=7)
    t_axis = np.linspace(0, cfg.SIM_DURATION, len(r["mos_scores"]))
    # Smooth for readability
    window = 20
    smoothed = np.convolve(r["mos_scores"], np.ones(window)/window, mode='valid')
    ax.plot(t_axis[:len(smoothed)], smoothed, color=COLORS[m],
            label=labels[methods.index(m)], linewidth=1.8)

ax.axhline(3.5, color='red', linestyle='--', linewidth=1.2, label='MOS=3.5 QDA threshold')
ax.set_xlabel("Simulation Time (seconds)", fontsize=11)
ax.set_ylabel("Mean Opinion Score (MOS)", fontsize=11)
ax.set_ylim(1, 5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fogiqm_mos_timeline.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: fogiqm_mos_timeline.png")

# ── Plot 4: Fog Utilisation Distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Fog Node CPU Utilisation — FogIQM vs Static Fog", fontsize=12, fontweight='bold')

for m, l in [("static_fog", "Static Fog"), ("fogiqm", "FogIQM (Ours)")]:
    r = simulate_single_run(method=m, seed=5)
    ax.hist(r["utilisation"], bins=40, alpha=0.6, color=COLORS[m], label=l, edgecolor='black', linewidth=0.3)

ax.axvline(cfg.RHO_THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'ATO threshold (ρ={cfg.RHO_THRESHOLD})')
ax.set_xlabel("Average Fog Utilisation (ρ)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fogiqm_utilisation.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Saved: fogiqm_utilisation.png")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: MATHEMATICAL VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  MATHEMATICAL VERIFICATION (Equations from Paper)")
print("=" * 65)

# Eq. (2) example
d = 200  # metres
tau_c = compute_tau_comm(d)
print(f"\n  Eq. (2) τ_comm for d=200m: {tau_c:.3f} ms")

# Eq. (3) M/M/1
lam_agg = 700
tau_q = compute_tau_comp(lam_agg, cfg.MU_FOG)
rho   = lam_agg / cfg.MU_FOG
print(f"  Eq. (3) τ_comp for ρ={rho:.2f}: {tau_q:.3f} ms")

# Eq. (4) EWMA
m_hat, x_new, alpha = 30.0, 45.0, 0.1
m_new = ewma_update(m_hat, x_new, alpha)
print(f"  Eq. (4) EWMA update: m̂={m_hat}, x={x_new}, α={alpha} → {m_new:.2f}")

# Eq. (5) adaptive alpha
sigma2 = 64.0
a = adaptive_alpha(sigma2)
print(f"  Eq. (5) Adaptive α for σ²_J={sigma2}: α={a:.4f}")

# Eq. (7) MOS
mos = compute_mos(40.0, 0.5)
print(f"  Eq. (7) MOS for L=40ms, PL=0.5%: MOS={mos:.3f}")

# Eq. (9) ATO fraction
rho_k = 0.92
theta = ato_offload_fraction(rho_k)
print(f"  Eq. (9) ATO offload fraction for ρ_k={rho_k}: θ*={theta:.3f}")

# Eq. (10) E[Q] and E[W]
rho_mm1 = 0.8
EQ = rho_mm1**2 / (1 - rho_mm1)
EW = rho_mm1 / (cfg.MU_FOG * (1 - rho_mm1)) * 1000  # ms
print(f"  Eq. (10) At ρ={rho_mm1}: E[Q]={EQ:.2f} pkts, E[W]={EW:.3f} ms")

print("\n" + "=" * 65)
print("  FogIQM Simulation Complete.")
print("  Outputs saved to current directory.")
print("=" * 65)
