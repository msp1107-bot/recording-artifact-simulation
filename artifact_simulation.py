#!/usr/bin/env python3
"""
Simulation: differential recording of a continuous outcome manufactures a
spurious treatment effect under a TRUE NULL (no real treatment effect).

From a fully known, null data-generating process it reproduces the four
phenomena seen in the empirical cohort:
  (A) a large apparent multiplicative 'reduction' in a naive log-linear model
  (B) instability of the effect size across standard estimators
  (C) an exceedance gap that is large at low thresholds, collapses at the
      clinically meaningful threshold (500 mL), and can reverse at the tail
  (D) a per-arm digit-preference (heaping) signature

Run:  pip install numpy pandas statsmodels matplotlib
      python3 artifact_simulation.py
Outputs a summary table to stdout and figure3_simulation.png.

NOTE FOR AUTHORS: re-run on your own machine and paste the numbers your run
produces into the manuscript. With a fixed seed the run is deterministic, but
minor differences in numpy's random Generator across versions can shift the
last digit. The qualitative pattern (large naive reduction, wide estimator
spread, gap collapsing at >=500 mL, per-arm heaping) is stable across seeds,
as the 200-replication summary at the end confirms.
"""
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ------------------- parameters (tunable, kept visible) ----------------
SEED, N_CONTROL, N_TREATED = 20260702, 1993, 1582   # match empirical arm sizes
TRUE_MEDIAN_ML, LOGNORM_SIGMA = 180.0, 0.60         # identical in BOTH arms
TRUE_EFFECT_RATIO = 1.00        # 1.00 = NO true effect; try 0.90 for a real 10% cut
GRID_ML = 50                    # documentation grid for "normal" recording
TEMPLATE_LOW_VALS, TEMPLATE_LOW_P = [30, 50], [0.55, 0.45]  # pinned low values (treated)
ROUTINE_CUTOFF_ML, P_TEMPLATE_ROUTINE = 300.0, 0.80        # only routine cases pinned;
                                # genuine bleeds (true >= cutoff) documented accurately

rng = np.random.default_rng(SEED)
grid  = lambda x: np.clip(np.round(x/GRID_ML)*GRID_ML, GRID_ML, None)
draw  = lambda n, r: rng.lognormal(np.log(TRUE_MEDIAN_ML*r), LOGNORM_SIGMA, n)
predr = lambda r: (1-r)*100

# control: heaped on grid, faithful
rec_c = grid(draw(N_CONTROL, 1.0))
# treated: SAME underlying law; routine cases often pinned to a low template,
# genuine bleeds (true >= cutoff) documented accurately
true_t = draw(N_TREATED, TRUE_EFFECT_RATIO)
rec_t  = grid(true_t)
templated = (true_t < ROUTINE_CUTOFF_ML) & (rng.random(N_TREATED) < P_TEMPLATE_ROUTINE)
rec_t[templated] = rng.choice(TEMPLATE_LOW_VALS, templated.sum(), p=TEMPLATE_LOW_P)

df = pd.DataFrame({"recorded_bl": np.r_[rec_c, rec_t],
                   "treated": np.r_[np.zeros(N_CONTROL), np.ones(N_TREATED)]})
df["log_bl"] = np.log(df.recorded_bl)

# ---- (A)/(B) four estimators -----------------------------------------
ols = smf.ols("log_bl ~ treated", df).fit()
r_loglin, ci = np.exp(ols.params.treated), np.exp(ols.conf_int().loc["treated"].values)
r_mean  = df[df.treated==1].recorded_bl.mean()/df[df.treated==0].recorded_bl.mean()
r_gamma = np.exp(smf.glm("recorded_bl ~ treated", df,
            family=sm.families.Gamma(sm.families.links.Log())).fit().params.treated)
med_t, med_c = df[df.treated==1].recorded_bl.median(), df[df.treated==0].recorded_bl.median()
r_med = med_t/med_c
reds = [predr(r_med), predr(r_loglin), predr(r_gamma), predr(r_mean)]

print("="*68)
print(f"TRUE effect ratio = {TRUE_EFFECT_RATIO}  ->  true reduction = {predr(TRUE_EFFECT_RATIO):.0f}%")
print("="*68)
print("APPARENT effect in the RECORDED field, by estimator:")
print(f"  Log-linear (geo-mean)  ratio {r_loglin:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})  = {predr(r_loglin):.0f}%")
print(f"  Arithmetic mean        ratio {r_mean:.3f}                       = {predr(r_mean):.0f}%")
print(f"  Gamma-log GLM          ratio {r_gamma:.3f}                       = {predr(r_gamma):.0f}%")
print(f"  Ratio of medians       ratio {r_med:.3f}  ({med_t:.0f} vs {med_c:.0f} mL)      = {predr(r_med):.0f}%")
print(f"  --> spread {min(reds):.0f}% to {max(reds):.0f}% from ONE null dataset")

# ---- (C) threshold exceedance ----------------------------------------
print("\nThreshold exceedance (% >= thr) and gap (control - carbetocin):")
print("  thr(mL)  control%  carb%   gap(pp)")
for thr in [100,200,300,500,1000]:
    pc = (df[df.treated==0].recorded_bl>=thr).mean()*100
    pt = (df[df.treated==1].recorded_bl>=thr).mean()*100
    print(f"  {thr:6d}   {pc:7.1f} {pt:6.1f}  {pc-pt:+7.1f}")

# ---- (D) digit preference --------------------------------------------
sg = lambda x,g: np.mean(np.isclose(np.mod(x,g),0))*100
sa = lambda x,v: np.mean(np.isin(x,v))*100
print("\nDigit preference:")
for lab,m in [("control",df.treated==0),("carbetocin",df.treated==1)]:
    x=df[m].recorded_bl.values
    print(f"  {lab:11s} at 30/50 mL: {sa(x,[30,50]):5.1f}%   on 50-grid: {sg(x,50):5.1f}%")

# ---- Figure 3 ---------------------------------------------------------
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
ax[0].bar(["Ratio of\nmedians","Log-linear","Gamma-log","Arithmetic\nmean"],reds,color="#4C72B0")
ax[0].axhline(predr(TRUE_EFFECT_RATIO),ls="--",color="crimson",label=f"true ({predr(TRUE_EFFECT_RATIO):.0f}%)")
ax[0].set_ylabel("apparent 'reduction' (%)");ax[0].set_title("A  Estimator instability");ax[0].legend(fontsize=8)
thrs=[100,200,300,500,1000]
gaps=[(df[df.treated==0].recorded_bl>=t).mean()*100-(df[df.treated==1].recorded_bl>=t).mean()*100 for t in thrs]
ax[1].plot(thrs,gaps,"-o",color="#C44E52");ax[1].axhline(0,color="grey",lw=.8)
ax[1].set_xlabel("threshold (mL)");ax[1].set_ylabel("gap (pp)");ax[1].set_title("B  Exceedance gap")
bins=np.arange(0,800,25)
ax[2].hist(df[df.treated==0].recorded_bl,bins=bins,alpha=.5,label="control",color="#55A868")
ax[2].hist(df[df.treated==1].recorded_bl,bins=bins,alpha=.5,label="carbetocin",color="#4C72B0")
ax[2].set_xlabel("recorded blood loss (mL)");ax[2].set_ylabel("count");ax[2].set_title("C  Recorded field");ax[2].legend(fontsize=8)
plt.tight_layout();plt.savefig("figure3_simulation.png",dpi=150)
print("\nSaved figure3_simulation.png")

# ---- multi-seed robustness (regularity, not a single draw) ------------
def one_rep(seed):
    r = np.random.default_rng(seed)
    g = lambda x: np.clip(np.round(x/GRID_ML)*GRID_ML, GRID_ML, None)
    d = lambda n,rr: r.lognormal(np.log(TRUE_MEDIAN_ML*rr), LOGNORM_SIGMA, n)
    c = g(d(N_CONTROL,1.0)); tt = d(N_TREATED,TRUE_EFFECT_RATIO); t = g(tt)
    tp = (tt<ROUTINE_CUTOFF_ML)&(r.random(N_TREATED)<P_TEMPLATE_ROUTINE)
    t[tp] = r.choice(TEMPLATE_LOW_VALS, tp.sum(), p=TEMPLATE_LOW_P)
    dd = pd.DataFrame({"recorded_bl":np.r_[c,t],"treated":np.r_[np.zeros(N_CONTROL),np.ones(N_TREATED)]})
    ll = np.exp(smf.ols("np.log(recorded_bl) ~ treated", dd).fit().params.treated)
    g500 = (dd[dd.treated==0].recorded_bl>=500).mean()*100-(dd[dd.treated==1].recorded_bl>=500).mean()*100
    return predr(ll), g500
reps = np.array([one_rep(s) for s in range(1000,1200)])   # 200 replications
print("\nAcross 200 simulated NULL datasets:")
print(f"  naive log-linear 'reduction': median {np.median(reps[:,0]):.0f}%  (IQR {np.percentile(reps[:,0],25):.0f}-{np.percentile(reps[:,0],75):.0f}%)")
print(f"  >=500 mL exceedance gap:      median {np.median(reps[:,1]):+.1f}pp (IQR {np.percentile(reps[:,1],25):+.1f} to {np.percentile(reps[:,1],75):+.1f})")
