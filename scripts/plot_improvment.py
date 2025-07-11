import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/improvement_vs_bruteforce.csv")

fig, ax = plt.subplots()
for p, grp in df.groupby("p"):
    ax.plot(grp["N"], grp["percent_gain"], marker="o", label=f"p={p}")

ax.set_xlabel("Assets N")
ax.set_ylabel("% runtime gain (brute → enhanced)")
ax.set_title("CPU speed-up vs pure-Python brute force")
ax.legend()
fig.tight_layout()
plt.show()
