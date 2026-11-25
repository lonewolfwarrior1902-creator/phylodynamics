import io
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from Bio import SeqIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment

st.set_page_config(page_title="Phylodynamics & SIR Simulator", layout="wide")
st.title("Viral Phylodynamics & Stochastic SIR Simulator")

# =====================================================================
# DEFAULT BUILT-IN DATA (Aligned Viral Strains & Collection Dates)
# =====================================================================
DEFAULT_FASTA = """>Wuhan_2019
ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACCAGAACT
>Alpha_2020
ATGTTGTTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACCAGAACT
>Beta_2021
ATGTTGTTTTTTCTTGTTTTATTGCCACTAGTCTCTAGCCAGTGTGTTAATCTTACAACCAGAACT
>Delta_2021
ATGTTGTTTTTTCTTGTTTTATTGCCACTAGTCTCTAGCCAGTGTGTTAACCTTACAACCAGAACT
>Omicron_BA1_2021
ATGTTGTTTTTTCTTGTTTTATTGCCACTAGTCTCTAGCCAGTGTGTTAACCTTACAATCAGAACT
>Omicron_BA2_2022
ATGTTGTTTTTTCTTGTTTTATTGCCACTAGTCTCTAGCCAGTGTGTTAACCTTACAATCAGACCT
>Omicron_XBB_2023
ATGTTGTTTTTTCTTGTTTTATTGCCACTAGTCTCTAGCCAGTGTGTTAACCTTACAATCAGATCT"""

DEFAULT_CSV = """sequence_id,collection_date
Wuhan_2019,2019-12-26
Alpha_2020,2020-09-20
Beta_2021,2021-03-10
Delta_2021,2021-05-18
Omicron_BA1_2021,2021-11-24
Omicron_BA2_2022,2022-02-14
Omicron_XBB_2023,2023-01-12"""

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("1. Data Source")
mode = st.sidebar.radio("Select Input Mode:", ["Default Dataset", "Manual Input"])

st.sidebar.header("2. SIR Epidemic Parameters")
N = st.sidebar.number_input("Population Size (N)", min_value=100, max_value=100000, value=10000, step=500)
beta = st.sidebar.slider("Transmission Rate (β)", 0.05, 1.0, 0.30, 0.01)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.01, 0.5, 0.10, 0.01)
t_max = st.sidebar.slider("Simulation Days", 20, 365, 100)

if mode == "Default Dataset":
    fasta_text, csv_text = DEFAULT_FASTA, DEFAULT_CSV
else:
    col_in1, col_in2 = st.columns(2)
    fasta_text = col_in1.text_area("Paste or Edit Aligned FASTA Sequences:", DEFAULT_FASTA, height=200)
    csv_text = col_in2.text_area("Paste or Edit Metadata CSV (YYYY-MM-DD):", DEFAULT_CSV, height=200)

# =====================================================================
# STEP 1: PARSE & CALCULATE JUKES-CANTOR DISTANCE MATRIX
# =====================================================================
try:
    records = list(SeqIO.parse(io.StringIO(fasta_text), "fasta"))
    alignment = MultipleSeqAlignment(records)
    df_meta = pd.read_csv(io.StringIO(csv_text))
except Exception as e:
    st.error(f"Input format error: {e}. Ensure all FASTA sequences have equal length.")
    st.stop()

if len(alignment) < 3:
    st.warning("Please provide at least 3 aligned FASTA sequences.")
    st.stop()

if not {"sequence_id", "collection_date"}.issubset(df_meta.columns):
    st.error("CSV metadata must contain 'sequence_id' and 'collection_date' columns.")
    st.stop()

dates_dict = dict(
    zip(
        df_meta["sequence_id"].astype(str).str.strip(),
        df_meta["collection_date"].astype(str).str.strip(),
    )
)

n = len(alignment)
seq_len = len(alignment[0].seq)
matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        s1, s2 = str(alignment[i].seq).upper(), str(alignment[j].seq).upper()
        diffs = sum(a != b for a, b in zip(s1, s2))
        p = min(diffs / seq_len, 0.749)
        matrix[i][j] = matrix[j][i] = max(0.0, -0.75 * np.log(1.0 - (4.0 / 3.0) * p))

lower_tri = [[matrix[i][j] for j in range(i + 1)] for i in range(n)]
dist_matrix = DistanceMatrix(names=[r.id for r in alignment], matrix=lower_tri)

# =====================================================================
# STEP 2: NEIGHBOR-JOINING TREE & MOLECULAR CLOCK REGRESSION
# =====================================================================
tree = DistanceTreeConstructor().nj(dist_matrix)

valid_dates = {}
for leaf in tree.get_terminals():
    date_str = dates_dict.get(leaf.name)
    if date_str:
        try:
            valid_dates[leaf.name] = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            st.error(f"Invalid date format for '{leaf.name}': '{date_str}'. Use YYYY-MM-DD.")
            st.stop()

if len(valid_dates) < 2:
    st.error("At least 2 sequence IDs must match between the FASTA headers and CSV metadata.")
    st.stop()

# Root tree at the earliest sampled strain (ancestral outgroup)
earliest_strain = min(valid_dates, key=valid_dates.get)
tree.root_with_outgroup(earliest_strain)

distances, times = [], []
for leaf in tree.get_terminals():
    if leaf.name in valid_dates:
        dt = valid_dates[leaf.name]
        times.append(dt.year + (dt.timetuple().tm_yday - 1) / 365.25)
        distances.append(tree.distance(tree.root, leaf))

times, distances = np.array(times), np.array(distances)
slope, intercept, r_val, _, _ = stats.linregress(times, distances)
t_mrca = -intercept / slope if slope != 0 else 0.0

# =====================================================================
# STEP 3: STOCHASTIC SIR SIMULATION (GILLESPIE ALGORITHM)
# =====================================================================
I0 = min(5, N - 1)
S, I, R, t = N - I0, I0, 0, 0.0
sim_t, S_list, I_list, R_list = [0.0], [S], [I], [R]
np.random.seed(42)

while t < t_max and I > 0:
    r_inf, r_rec = beta * S * I / N, gamma * I
    r_tot = r_inf + r_rec
    if r_tot <= 0:
        break
    t += np.random.exponential(1.0 / r_tot)
    if np.random.rand() < (r_inf / r_tot):
        S, I = S - 1, I + 1
    else:
        I, R = I - 1, R + 1
    sim_t.append(t)
    S_list.append(S)
    I_list.append(I)
    R_list.append(R)

# Downsample trajectory points if N is huge to keep Matplotlib fast & clean
if len(sim_t) > 3000:
    step = len(sim_t) // 3000
    sim_t = sim_t[::step] + [sim_t[-1]]
    S_list = S_list[::step] + [S_list[-1]]
    I_list = I_list[::step] + [I_list[-1]]
    R_list = R_list[::step] + [R_list[-1]]

# =====================================================================
# STEP 4: DISPLAY METRICS & VISUALIZATIONS
# =====================================================================
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Substitution Rate (μ)", f"{slope:.5f} subs/site/yr")
c2.metric("Estimated tMRCA", f"{t_mrca:.2f}")
c3.metric("Clock Fit (R²)", f"{r_val**2:.4f}")
c4.metric("Peak Infected", f"{max(I_list):,}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")

# Plot 1: Molecular Clock Regression
axes[0].scatter(times, distances, color="dodgerblue", label="Viral Samples", zorder=3)
axes[0].plot(times, slope * times + intercept, "r--", lw=2, label=rf"Fit ($\mu$={slope:.4f})")
axes[0].set(title="Root-to-Tip Molecular Clock", xlabel="Collection Date (Year)", ylabel="Root-to-Tip Distance")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Stochastic SIR Outbreak Trajectory
axes[1].plot(sim_t, S_list, label="Susceptible S(t)", color="blue", alpha=0.75)
axes[1].plot(sim_t, I_list, label="Infected I(t)", color="crimson", lw=2.2)
axes[1].plot(sim_t, R_list, label="Recovered R(t)", color="green", alpha=0.75)
axes[1].set(
    title=f"Stochastic SIR Trajectory ($R_0$ = {beta/gamma:.2f})",
    xlabel="Time (Days)",
    ylabel="Population",
    xlim=(0, t_max),
    ylim=(0, N * 1.05),
)
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

# Plot 3: Neighbor-Joining Phylogenetic Tree
Phylo.draw(tree, axes=axes[2], do_show=False)
axes[2].set_title("Phylogenetic Tree (NJ)")

st.pyplot(fig)
plt.close(fig)
