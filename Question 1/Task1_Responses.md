# Task 1 — Responses

All numbers come from the `PA1_PRESET=full` run (data seed 0, model seed 0). Unless stated
otherwise, errors are **validation-region** RMSE in m/s² (Outputs 1.1, 2.3, 4.1, 4.2). "Established"
= Stations 1–3, "held-out" = Station 4. Figures: `results/design/<output>-1.pdf`.

> **Note to self before submitting:** the two "before running" parts (the prior ordering in 1A
> and the prediction in 4a) must be what *I* expected. Edit them if my expectation differed.

---

## Response 1A — Predict, measure, and revise the aggregate ordering

**Prior ordering (recorded before Output 1.1).**
Established: Raw Attention ≈ Period-routed ridge < Sensor-specific ridge ≈ Shared ridge.
Held-out Station 4: Raw Attention < Period-routed ridge < Shared ridge (Sensor-specific has no fit).
Reasoning: Attention is the only model that adapts its rule to the context. Station identity does
not determine pace, so a per-station matrix should not beat the shared one.

**Measurements (Output 1.1, validation RMSE m/s²).**

| Model | Established | A | B | C | Station 4 | Params |
|---|---|---|---|---|---|---|
| Shared ridge | 0.767 | 0.805 | 0.854 | 0.621 | 0.876 | 4,608 |
| Sensor-specific ridge | 0.768 | 0.804 | 0.852 | 0.630 | unavailable | 13,824 |
| Period-routed ridge | **0.166** | 0.181 | 0.164 | 0.152 | **0.173** | 59,904 |
| Raw Attention | 0.421 | 0.380 | 0.475 | 0.403 | 0.608 | 368,368 |

**Key differences.**
- Routing on period cuts the shared-ridge error by **78 %** (0.767 → 0.166). Station 4 shows the same cut (0.876 → 0.173).
- A matrix per station gives **no** gain (0.768 vs 0.767). It triples the parameters and cannot be used on Station 4.
- Raw Attention beats both non-routed ridges (−45 % vs shared). It is still **2.5×** worse than period-routed ridge, with 6× more parameters and ~800 s of training instead of milliseconds.
- **Generalisation gap.**
  - Period-routed ridge degrades only 4 % on Station 4 (0.166 → 0.173).
  - Raw Attention degrades 44 % (0.421 → 0.608).
  - Shared ridge degrades 14 % (0.767 → 0.876).
- **Per product.**
  - The shared ridge is worst on the fast products A and B (0.805, 0.854) and best on C (0.621).
  - Period-routed ridge is flat across products (0.152–0.181).

**Revised ordering.** For both populations: Period-routed ridge < Raw Attention < Shared ridge ≈
Sensor-specific ridge. Sensor-specific is unavailable for Station 4.

**Explanation via assumptions.**
- **Shared ridge** assumes one map fits every window. Phase advance depends on pace (14–36 samples), so one operator must compromise.
- **Sensor-specific ridge** assumes windows differ by station. They don't: every station experiences every pace. Each per-station matrix faces the same compromise, hence identical error.
- **Period-routed ridge** assumes windows differ by pace, which matches the generator. Within one 1.5-sample pace band the right continuation is nearly linear, so a linear map per band is close to optimal. It conditions on pace, not identity, so it transfers to Station 4.
- **Raw Attention** assumes the comparison rule can be learned from the window itself. That is true in principle, but it must discover the period from data while also modelling the slow ramp. Its pairwise scores also have no notion that all pairs at the same lag matter for the same reason. It fits the established stations' amplitude, phase and drift more than the pace. That is why its Station 4 error rises the most.

---

## Response 1B — Why the forecasting rule must depend on context

**The compromise in Ŵ (Output 1.2, Station 1).**
Window RMSE for the Product A run (P = 16.1 samples):
- shared ridge 0.779
- period-routed ridge 0.232
- Raw Attention 0.288

For the Product C run (P = 30.1): 0.448, 0.150 and 0.205 respectively.

Ŵ = (XᵀX/n + λI)⁻¹XᵀZ/n is the least-squares solution over *all* training windows. A context with
period 16 and one with period 30 need different phase advances over the next 48 samples. Ŵ has no
input that selects which advance applies. It therefore returns roughly the *average* continuation
over paces.

Averaging sinusoids with different phases shrinks their amplitude, and that is what the figure
shows:
- **Product A:** the shared-ridge forecast keeps the right timing for the first peak but is visibly damped. It peaks at ~5.0 vs a true 6.2 and bottoms at ~3.5 vs a true 2.4, and the oscillation flattens further out.
- **Product C:** it flattens the second hump (~0.9 vs a true 1.8).

The compromise is therefore an **amplitude/shape shrinkage toward the pace-averaged
continuation**, growing with horizon. Sensor-specific ridge produces the same damped curve,
because each station's training set mixes all paces too.

**Same projections, different mixing (Output 1.3).** The same Station 1 sensor and the same frozen
W_Q, W_K, W_V give different attention maps for the A and C contexts:
- The A map concentrates weight on one key column (~69) and the last ~10 keys.
- The C map spreads weight over bands near keys 40–50, 65 and 85–95.

The A − C difference has a maximum of |0.52| and a mean of 0.011. The weights come from
softmax(QKᵀ/√d_h) with Q = XW_Q and K = XW_K, so changing X changes QKᵀ and therefore which values
are mixed. The *parameters* are fixed, but the *value-mixing operator* is rebuilt per context
without refitting or a router.

Ridge is the opposite. Its operator Ŵ is fixed once, and only the input passed through it changes.
This only shows that Attention's mixing depends on the context; it does not show that Attention
recovered the physical period.

---

## Response 2 — Separate slow structure from repetition

**Selected width: m = 41.** It was chosen automatically by the training-only oracle diagnostic
(Output 2.2).

**Visual change (Output 2.1, Station 2, Product C, P = 32).**
- With m = 17 the "trend" still follows part of the vibration (wavy). The remainder is damped: residual std 0.58 m/s².
- As m grows, the trend becomes a smooth ramp (range stays ~8 m/s²). The remainder becomes a clean, full-amplitude, near-sinusoidal cycle: std rises to 1.01, 1.37 and 1.57 m/s² for m = 25, 33 and 41.
- Near the two boundaries the replicate padding flattens the trend (visible at both ends of the panel), so the first and last ~20 remainder values carry some leftover ramp.
- The right panel shows the effect on pace. The period-projection score of the raw context is almost flat (~3). After decomposition it has a sharp peak (~45) at the true period of 32.

**Empirical change (Output 2.2, established training windows).** Share of contexts whose period
estimate falls within one sample of the true period:

| Representation | Within one sample | Median abs. period error (samples) |
|---|---|---|
| raw | 0.687 | 0.52 |
| m = 17 | 0.980 | 0.16 |
| m = 25 | 0.989 | 0.12 |
| m = 33 | 0.993 | 0.14 |
| m = 41 | **0.998** | 0.22 |

m = 41 maximises the within-one-sample recovery. m = 25 has a slightly lower median error, so the
choice trades a few rare large misses against small typical errors.

**Validation comparison (Output 2.3).**

| | Established | Station 4 |
|---|---|---|
| Raw Attention | 0.421 | 0.608 |
| Attention + decomposition | 0.287 (−32 %) | 0.428 (−30 %) |

The gain holds for every product:
- established A/B/C: 0.380/0.475/0.403 → 0.261/0.323/0.273
- Station 4 A/B/C: 0.639/0.571/0.612 → 0.350/0.481/0.442

The largest per-window gain is Station 3, Product B, whose context has a steep downward ramp. Raw
Attention drifts out of phase after ~1 s there, while the decomposed model stays aligned. The
smallest gain is Station 4, Product B, whose context is nearly flat, so there is little drift to
remove.

**Why a moving average is a sensible bias here.**
- The world is drift plus repetition with **separated time scales**: drift varies over ~240 samples and vibration over 14–36.
- A centered average with a window longer than the period cancels most of the cycle while passing the slow part. Inside 96 samples the slow part is close to linear, and a symmetric moving average reproduces linear trends exactly.
- The linear trend head then handles the ramp, and the encoder sees a nearly zero-mean periodic signal.
- The decomposed forecaster is a fixed, parameter-free filter, so it cannot overfit station-specific drift. That is consistent with the 30 % gain on held-out Station 4.

**Why the training-only oracle diagnostic is a good way to choose m.**
- m is the one design choice, and the quantity it should serve (how well the remainder exposes the repetition) is directly measurable in this synthetic world.
- The generator's true period is used only to *score* candidate widths, only on established-station training windows. It never enters the model, the routing, or validation/test.
- So no leakage occurs, and the choice does not depend on training four neural models per width.

**Where this bias fails / an alternative.**
- **Failure:** an abrupt level shift, for example a calibration jump inside the context. A centered average smears the step into a ramp ±20 samples wide, leaving a spurious bump in S that looks like vibration. Near the context edges, the replicate padding also gets the slope wrong.
- **Alternative: STL / LOESS decomposition, or a robust median filter.**
  - LOESS fits a local polynomial, so it follows curvature and edges better.
  - A median filter preserves step changes.
  - Either encodes "trend is locally smooth but may bend or jump" instead of "trend is locally linear and well separated in frequency from the cycle". The cost is more hyperparameters and a less smooth (median) or more expensive (LOESS) operator.

---

## Response 3 — Mix by temporal delay

**Implementation check (Output 3.1 and the aggregation example).**
- `delay_scores` matches the direct circular correlation on q = k = [1, 3, 1, 3]: scores 4, −4, 4, −4 for τ = 0..3.
- `aggregate_delays` reproduces the worked example. With v = [10, 20, 30, 40], delays [1, 3] and weights [¾, ¼] it gives z = [35, 15, 25, 25], e.g. z₀ = ¾·v₃ + ¼·v₁ = 30 + 5 = 35.
- This confirms the index rule z_t = Σ α_j v_{(t−τ_j) mod L}: each delay looks **back** τ samples, wrapping circularly.
- All notebook checks pass, including the gradient check through both values and weights.

**Strongest lags (Output 3.2, Station 2, Product A, P = 16.1).** The residual-signal correlation
peaks at **τ = 16 (0.896)** and **τ = 32 (0.897)**, the integers nearest to P and 2P. Troughs fall
near τ ≈ 24 and 40, the half-period offsets P/2 + kP, where peaks align with troughs. The curve
rises again toward 48 ≈ 3P.

Multiples of P score as high as P itself because a periodic residual recurs at every integer
multiple of its period. This is why a top-K selection naturally collects P, 2P and 3P, and why
blending them is safe.

**Why decomposition cleans the delay evidence.**
- The raw-signal correlation is dominated by the ramp. It is highest at small lags (~0.45 at τ = 8) and decays monotonically to about −0.55 near τ = 40, with almost no bump at P or 2P.
- A rising line correlates with itself after any short shift, and circular wrap-around makes long shifts anti-correlated (the end of a ramp is aligned with its start).
- Removing the slow component leaves a near-zero-mean periodic signal, for which correlation is large only when peaks align. The peaks at kP become sharp, close to 0.9.
- This supports the delay mixer's bias: once drift is removed, the useful structure really is a small set of displacements (P and its multiples) shared by every position. Scoring L delays and reusing them at all positions is the right parameterisation. Scoring L² position pairs independently would have to rediscover the same lag at every position.

**When the delay bias is wrong.**
- **Example: a period that drifts quickly within one context**, such as a ramp-up from 14 to 20 samples over 96 samples.
- No single τ aligns peaks across the whole window. The circular correlation smears into a broad, low peak. Top-K picks compromise lags that are right for only part of the window, and the aggregated value z_t shifts copies that are out of phase at the window's ends.
- The weighted blend then cancels the oscillation, giving an amplitude-damped forecast: the same failure as the shared ridge, now inside the mixer.
- **Isolated transients and aperiodic events break it too.** A spike has no recurring displacement, and circular shifting copies the spike to every position τ_j later, injecting false echoes.
- Circularity also assumes sample L−1 sits next to sample 0, which is wrong whenever the window doesn't contain a whole number of periods.

---

## Response 4 — Compare designs and deploy

**(a) Prediction (written before Output 4.1).** Decomposition should help most on contexts with a
prominent slow component, because it removes the ramp before the mixer sees it and gives that ramp
to a linear trend head. The delay mixer alone still scores recurrence on a ramp that is
self-similar at every shift.

**(b) Validation results (Outputs 4.1, 4.2; established stations unless noted).**

| Model | RMSE | MSE | Station 4 RMSE | Station 4 MSE | Params | Fit (s) |
|---|---|---|---|---|---|---|
| Shared ridge | 0.767 | 0.588 | 0.876 | 0.767 | 4,608 | 0.002 |
| Sensor-specific ridge | 0.768 | 0.590 | — | — | 13,824 | 0.001 |
| **Period-routed ridge** | **0.166** | **0.028** | **0.173** | **0.030** | 59,904 | 0.002 |
| Raw Attention | 0.421 | 0.178 | 0.608 | 0.369 | 368,368 | 802 |
| Raw delay mixer | 0.224 | 0.050 | 0.337 | 0.113 | 368,370 | 1097 |
| Attention + decomposition | 0.287 | 0.082 | 0.428 | 0.183 | 373,024 | 967 |
| Autoformer-inspired | 0.198 | 0.039 | 0.240 | 0.057 | 373,026 | 1017 |

*Switches.* MSE reduction relative to Raw Attention:

| | Established | Station 4 |
|---|---|---|
| Delay mixing alone | 0.128 | 0.256 |
| Decomposition alone | 0.096 | 0.186 |
| Both (Autoformer-inspired) | 0.139 | 0.312 |
| Sum of the two individual gains | 0.224 | 0.442 |

- Each switch helps on its own, and delay mixing is the larger single upgrade in both populations.
- The combined gain is far **less than the sum** of the individual gains. It is only slightly more than delay mixing alone (0.139 vs 0.128 established).
- The upgrades **overlap rather than stack**. Both target the same drift-plus-repetition structure, and the delay mixer already gets much of the benefit of detrending implicitly: once it picks P-multiples, the ramp contributes little to the aggregated values.
- Combining helps more on Station 4 (RMSE 0.337 → 0.240) than on established stations (0.224 → 0.198).
- Prediction (a) was only partly supported. Decomposition's biggest window gain was on a steep-ramp context (Output 2.3, Station 3 B), but overall delay mixing helped more than decomposition.

*Horizon (Output 4.2, established RMSE, h = 1 → 24 → 48).*

| Model | h = 1 | h = 24 | h = 48 |
|---|---|---|---|
| Raw Attention | 0.149 | 0.432 | 0.530 |
| Attention + decomposition | 0.147 | 0.264 | 0.390 |
| Raw delay mixer | 0.119 | 0.188 | 0.351 |
| Autoformer-inspired | 0.135 | 0.181 | 0.235 |
| Period-routed ridge | 0.112 | 0.147 | 0.228 |

- All models are similar at h = 1, which is mostly continuity of the last sample.
- Errors diverge with horizon, because phase errors accumulate.
- Autoformer-inspired grows slowest among the neural models. On Station 4 the raw delay mixer degrades badly late in the horizon (0.138 → 0.619), while Autoformer-inspired stays at 0.144 → 0.304.
- Period-routed ridge is lowest or tied at every horizon in both populations (Station 4: 0.119 → 0.212).

**(c) Deployment choice (made from validation only, before Output 4.3).**
- **Established stations: Period-routed ridge.**
  - Validation RMSE 0.166 vs 0.198 for the best neural model, 16 % lower (MSE 0.028 vs 0.039).
  - It has 6× fewer parameters (59,904 vs 373,026) and fits in 0.002 s vs 1017 s.
  - It is better on every product and at every horizon.
- **Held-out Station 4: Period-routed ridge.**
  - Validation RMSE 0.173 vs 0.240, 28 % lower.
  - It conditions on pace, which is shared by all stations, rather than on identity. So it needs nothing from Station 4 except the context, and it loses only 4 % going from established to held-out stations.
  - Sensor-specific ridge is not eligible.
- **Caveat:** routing relies on a hand-picked period estimator and pace bands covering the paces seen in training. A new product with an unseen pace would fall outside the bands. Autoformer-inspired, the best context-adaptive model, is the fallback in that case.
- **Output 4.3 confirms the choice:** test RMSE is 0.161 (established) and 0.183 (Station 4).

**Interpretation (≤ 200 words).**
The right inductive bias mattered far more than flexibility. Period-routed ridge has one fitted
linear map per pace band and 6× fewer parameters, yet it beat every Attention-based variant on
both populations. It encodes the true generative fact, that pace selects the continuation,
explicitly.

The two neural upgrades show the same lesson inside the network:
- Decomposition (a fixed moving average) and delay mixing (structure organised by displacement) each removed 30–50 % of Raw Attention's error at essentially zero parameter cost.
- Combining them gave Autoformer-inspired, the best neural model (0.198 / 0.240).
- Their gains overlapped rather than stacked, because both target the same drift-plus-repetition structure.

Flexibility without the right bias was costly. Raw Attention had the most freedom but the largest
generalisation gap (+44 % on Station 4), which suggests it fitted station-specific features rather
than pace.

The advantage of learned models appears only where the hand-built bias breaks. That would happen
with paces outside the routing bands, or with structure no person thought to route on. That is
where Autoformer-inspired would be the fallback.
