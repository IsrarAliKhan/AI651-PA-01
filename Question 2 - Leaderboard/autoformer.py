import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent
DATA = HERE / "Data"
OUT = HERE / "results"

PRED_LEN = 168
SEQ_LEN = 168
LABEL_LEN = 48
PERIOD = 24
VAL_LEN = 26 * PRED_LEN
VAL_STRIDE = 24
CONTINUOUS = ["feature_A", "feature_B", "feature_C", "feature_D", "feature_E", "feature_F"]
BINARY = ["feature_G", "feature_H", "feature_I", "feature_J"]


# ----------------------------------------------------------------------------- model
class SeriesDecomp(nn.Module):
    """Moving-average trend with replicate padding; returns (seasonal, trend)."""
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):                       # [B,L,C]
        q = (self.kernel - 1) // 2
        padded = torch.cat([x[:, :1].expand(-1, q, -1), x, x[:, -1:].expand(-1, q, -1)], 1)
        trend = F.avg_pool1d(padded.transpose(1, 2), self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class SeasonalLayerNorm(nn.Module):
    """LayerNorm followed by removing the per-series time mean (thuml `my_Layernorm`)."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(1, keepdim=True)


class AutoCorrelation(nn.Module):
    """Period-based dependency discovery (FFT autocorrelation) + time-delay aggregation."""
    def __init__(self, d, heads, factor=1):
        super().__init__()
        self.heads, self.factor = heads, factor
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, queries, keys, values):   # [B,L,d], [B,S,d], [B,S,d]
        B, L, d = queries.shape
        S = keys.shape[1]
        h = self.heads
        q = self.q(queries).view(B, L, h, -1)
        k = self.k(keys).view(B, S, h, -1)
        v = self.v(values).view(B, S, h, -1)
        if L > S:                               # cross-correlation: pad keys/values to L
            k = torch.cat([k, torch.zeros_like(q[:, :L - S])], 1)
            v = torch.cat([v, torch.zeros_like(q[:, :L - S])], 1)
        else:
            k, v = k[:, :L], v[:, :L]
        q, k, v = (t.permute(0, 2, 3, 1) for t in (q, k, v))     # [B,h,E,L]
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=-1) * torch.fft.rfft(k, dim=-1).conj(),
                               n=L, dim=-1)                       # R(tau), [B,h,E,L]
        top_k = max(1, int(self.factor * math.log(L)))
        score = corr.mean((1, 2))                                 # [B,L]
        weights, delays = score.topk(top_k, -1)
        weights = weights.softmax(-1)
        index = (torch.arange(L, device=q.device) + delays[..., None]) % L   # roll by +tau
        out = torch.zeros_like(v)
        for j in range(top_k):
            out = out + weights[:, j, None, None, None] * v.gather(
                -1, index[:, None, None, j].expand_as(v))
        return self.o(out.permute(0, 3, 1, 2).reshape(B, L, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, d_ff, kernel, dropout):
        super().__init__()
        self.attn = AutoCorrelation(d, heads)
        self.ff = nn.Sequential(nn.Conv1d(d, d_ff, 1, bias=False), nn.GELU(), nn.Dropout(dropout),
                                nn.Conv1d(d_ff, d, 1, bias=False))
        self.decomp1, self.decomp2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.drop(self.attn(x, x, x)))
        y = self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2))
        return self.decomp2(x + y)[0]


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, d_ff, kernel, dropout, c_out):
        super().__init__()
        self.self_attn, self.cross_attn = AutoCorrelation(d, heads), AutoCorrelation(d, heads)
        self.ff = nn.Sequential(nn.Conv1d(d, d_ff, 1, bias=False), nn.GELU(), nn.Dropout(dropout),
                                nn.Conv1d(d_ff, d, 1, bias=False))
        self.decomps = nn.ModuleList(SeriesDecomp(kernel) for _ in range(3))
        self.trend_proj = nn.Conv1d(d, c_out, 3, padding=1, padding_mode="circular", bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross):
        x, t1 = self.decomps[0](x + self.drop(self.self_attn(x, x, x)))
        x, t2 = self.decomps[1](x + self.drop(self.cross_attn(x, cross, cross)))
        y = self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, t3 = self.decomps[2](x + y)
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class Embedding(nn.Module):
    """Value token conv + linear embedding of marks (phase and optional covariates)."""
    def __init__(self, c_in, n_marks, d, dropout):
        super().__init__()
        self.token = nn.Conv1d(c_in, d, 3, padding=1, padding_mode="circular", bias=False)
        self.marks = nn.Linear(n_marks, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, marks):
        return self.drop(self.token(x.transpose(1, 2)).transpose(1, 2) + self.marks(marks))


class Autoformer(nn.Module):
    def __init__(self, n_marks, d=32, heads=4, d_ff=64, enc_layers=1, dec_layers=1,
                 kernel=25, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomp(kernel)
        self.enc_embed = Embedding(1, n_marks, d, dropout)
        self.dec_embed = Embedding(1, n_marks, d, dropout)
        self.encoder = nn.ModuleList(EncoderLayer(d, heads, d_ff, kernel, dropout)
                                     for _ in range(enc_layers))
        self.enc_norm = SeasonalLayerNorm(d)
        self.decoder = nn.ModuleList(DecoderLayer(d, heads, d_ff, kernel, dropout, 1)
                                     for _ in range(dec_layers))
        self.dec_norm = SeasonalLayerNorm(d)
        self.projection = nn.Linear(d, 1)

    def forward(self, x_enc, mark_enc, mark_dec):   # [B,S,1], [B,S,M], [B,label+pred,M]
        mean = x_enc.mean(1, keepdim=True).expand(-1, PRED_LEN, -1)
        seasonal, trend = self.decomp(x_enc)
        trend = torch.cat([trend[:, -LABEL_LEN:], mean], 1)
        seasonal = torch.cat([seasonal[:, -LABEL_LEN:], torch.zeros_like(mean)], 1)

        enc = self.enc_embed(x_enc, mark_enc)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        dec = self.dec_embed(seasonal, mark_dec)
        for layer in self.decoder:
            dec, residual_trend = layer(dec, enc)
            trend = trend + residual_trend
        seasonal = self.projection(self.dec_norm(dec))
        return (trend + seasonal)[:, -PRED_LEN:, 0]    # [B,pred]


# ------------------------------------------------------------------------------ data
def load():
    train = pd.read_csv(DATA / "student_train.csv")
    ext = pd.read_csv(DATA / "optional_external_data.csv")
    assert (train.time_idx.values == np.arange(1, len(train) + 1)).all()
    assert len(ext) == len(train) + PRED_LEN
    return train.value.values.astype(np.float64), ext


def build_marks(ext, ext_mode, fit_end):
    """Marks per time step. ext_mode: none | past | future.
    'past' masks covariates beyond the forecast origin per window (done in make_batch)."""
    idx = ext.time_idx.values
    phase = 2 * np.pi * (idx % PERIOD) / PERIOD
    cols = [np.sin(phase), np.cos(phase)]
    if ext_mode != "none":
        cont = ext[CONTINUOUS].values.astype(np.float64)
        cont[:, 3] = np.log1p(cont[:, 3])   # cumulated wind speed is heavy-tailed
        cont[:, 4:] = np.log1p(cont[:, 4:])
        mu, sd = cont[:fit_end].mean(0), cont[:fit_end].std(0) + 1e-8   # training stats only
        cols += list(((cont - mu) / sd).T) + list(ext[BINARY].values.astype(np.float64).T)
    return np.stack(cols, 1).astype(np.float32)


class Scaler:
    def __init__(self, y, log):
        self.log = log
        z = np.log1p(y) if log else y
        self.mu, self.sd = z.mean(), z.std()

    def fwd(self, y):
        return ((np.log1p(y) if self.log else y) - self.mu) / self.sd

    def inv(self, z):
        z = z * self.sd + self.mu
        return np.clip(np.expm1(z) if self.log else z, 0, None)


def make_batch(series, marks, origins, ext_mode, device):
    """origins = index of first forecast step (0-based). Returns enc/dec tensors and targets."""
    o = torch.as_tensor(origins)[:, None]
    enc_idx = o - SEQ_LEN + torch.arange(SEQ_LEN)
    dec_idx = o - LABEL_LEN + torch.arange(LABEL_LEN + PRED_LEN)
    x = series[enc_idx][..., None]
    m_enc, m_dec = marks[enc_idx], marks[dec_idx].clone()
    if ext_mode == "past":                       # covariates known only up to the origin
        m_dec[:, LABEL_LEN:, 2:] = 0.0
    tgt_idx = dec_idx[:, LABEL_LEN:].clamp(max=len(series) - 1)
    return (x.to(device), m_enc.to(device), m_dec.to(device)), series[tgt_idx].to(device)


# -------------------------------------------------------------------------- training
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, series, marks, origins, ext_mode, device, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(origins), batch):
            inp, _ = make_batch(series, marks, origins[i:i + batch], ext_mode, device)
            out.append(model(*inp).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    return dict(rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))),
                smape=float(100 * np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-8))))


def run(cfg, seed, final=False, device="cpu", verbose=True):
    """Train one model. Validation run: fit on [0, N-VAL_LEN), early stop on validation origins.
    Final run: same recipe, train origins extended to the full history, fixed epoch count."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    y, ext = load()
    N = len(y)
    fit_end = N if final else N - VAL_LEN
    scaler = Scaler(y[:fit_end], cfg["log"])
    series = torch.as_tensor(scaler.fwd(np.concatenate([y, np.zeros(PRED_LEN)])), dtype=torch.float32)
    marks = torch.as_tensor(build_marks(ext, cfg["ext"], fit_end))
    train_origins = np.arange(SEQ_LEN, fit_end - PRED_LEN + 1)
    val_origins = np.arange(N - VAL_LEN, N - PRED_LEN + 1, VAL_STRIDE)
    y_val = np.stack([y[o:o + PRED_LEN] for o in val_origins])

    model = Autoformer(marks.shape[1], d=cfg["d"], heads=cfg["heads"], d_ff=cfg["d_ff"],
                       kernel=cfg["kernel"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    epochs = cfg["epochs"] if final else cfg["max_epochs"]
    best, best_state, best_epoch, bad, history = np.inf, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        # one epoch = every train_stride-th window, fresh random offset each epoch
        offset = np.random.randint(cfg["train_stride"])
        perm = np.random.permutation(train_origins[offset::cfg["train_stride"]])
        for g in opt.param_groups:               # halve lr each epoch (thuml "type1" schedule)
            g["lr"] = cfg["lr"] * 0.5 ** (epoch - 1)
        for i in range(0, len(perm), cfg["batch"]):
            inp, tgt = make_batch(series, marks, perm[i:i + cfg["batch"]], cfg["ext"], device)
            loss = F.mse_loss(model(*inp), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if final:
            continue
        pred = scaler.inv(predict(model, series, marks, val_origins, cfg["ext"], device))
        m = metrics(y_val, pred)
        history.append(dict(epoch=epoch, **m))
        if verbose:
            print(f"  seed {seed} epoch {epoch}: val RMSE {m['rmse']:.2f} MAE {m['mae']:.2f} "
                  f"sMAPE {m['smape']:.1f} ({time.time() - t0:.0f}s)", flush=True)
        if m["rmse"] < best:
            best, best_epoch, bad = m["rmse"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if not final:
        model.load_state_dict(best_state)
    return model, dict(seed=seed, best_epoch=best_epoch, epochs_run=epoch, history=history,
                       params=count_params(model), seconds=time.time() - t0), (series, marks, scaler)


def baselines():
    y, _ = load()
    N = len(y)
    origins = np.arange(N - VAL_LEN, N - PRED_LEN + 1, VAL_STRIDE)
    truth = np.stack([y[o:o + PRED_LEN] for o in origins])
    persistence = np.stack([np.full(PRED_LEN, y[o - 1]) for o in origins])
    daily = np.stack([np.tile(y[o - PERIOD:o], PRED_LEN // PERIOD) for o in origins])
    mean_day = np.stack([np.full(PRED_LEN, y[o - PERIOD:o].mean()) for o in origins])
    return {"persistence": metrics(truth, persistence), "seasonal naive (24)": metrics(truth, daily),
            "last-24h mean": metrics(truth, mean_day)}


BASE = dict(d=32, heads=4, d_ff=64, kernel=25, dropout=0.1, lr=1e-3, batch=64,
            max_epochs=8, patience=2, train_stride=4, log=False, ext="none")


def ablate(seeds, device):
    OUT.mkdir(exist_ok=True)
    configs = {
        "no ext, raw target": dict(BASE),
        "no ext, log target": dict(BASE, log=True),
        "ext past-only, raw target": dict(BASE, ext="past"),
        "ext future, raw target": dict(BASE, ext="future"),
        "ext future, log target": dict(BASE, ext="future", log=True),
    }
    rows = [dict(config=k, seed="-", **v) for k, v in baselines().items()]
    print(pd.DataFrame(rows).to_string(), flush=True)
    for name, cfg in configs.items():
        for seed in seeds:
            print(f"{name} | seed {seed}", flush=True)
            _, info, _ = run(cfg, seed, device=device)
            best = info["history"][info["best_epoch"] - 1]
            rows.append(dict(config=name, seed=seed, rmse=best["rmse"], mae=best["mae"],
                             smape=best["smape"], best_epoch=info["best_epoch"],
                             epochs_run=info["epochs_run"], params=info["params"],
                             seconds=round(info["seconds"])))
            pd.DataFrame(rows).to_csv(OUT / "ablation_runs.csv", index=False)
    df = pd.DataFrame(rows)
    summary = (df[df.seed != "-"].groupby("config", sort=False)
               .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                    mae_mean=("mae", "mean"), smape_mean=("smape", "mean"),
                    best_epoch_mean=("best_epoch", "mean"), params=("params", "first")))
    summary.to_csv(OUT / "ablation_summary.csv")
    print(summary.round(2).to_string())


def final(args, device):
    """Train the chosen config on the full history for a fixed epoch count; write submission."""
    OUT.mkdir(exist_ok=True)
    cfg = dict(BASE, ext=args.ext, log=args.log, epochs=args.epochs)
    model, info, (series, marks, scaler) = run(cfg, args.seed, final=True, device=device)
    N = len(load()[0])
    pred = scaler.inv(predict(model, series, marks, np.array([N]), cfg["ext"], device))[0]
    assert pred.shape == (PRED_LEN,) and np.isfinite(pred).all()
    line = ", ".join(f"{v:.2f}" for v in pred)
    (OUT / "submission.txt").write_text(line + "\n")
    pd.DataFrame({"time_idx": np.arange(N + 1, N + PRED_LEN + 1), "value": pred}).to_csv(
        OUT / "submission.csv", index=False)
    meta = dict(cfg, seed=args.seed, trainable_params=info["params"], epochs=args.epochs)
    (OUT / "submission_meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), OUT / "final_model.pt")
    print(json.dumps(meta, indent=2))
    print(f"P = {info['params']} trainable parameters, E = {args.epochs} epochs")


def figures():
    """Report figures: periodicity evidence, ablation over seeds, one validation forecast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    y, ext = load()
    N = len(y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    lags = np.arange(1, 24 * 8 + 1)
    acf = [np.corrcoef(y[:-k], y[k:])[0, 1] for k in lags]
    axes[0].plot(lags, acf, color="#2a6f97")
    for k in range(24, lags[-1] + 1, 24):
        axes[0].axvline(k, color="#bbb", lw=0.6, ls=":")
    axes[0].set(xlabel="lag (steps)", ylabel="autocorrelation", title="ACF: fast decay, daily ripple")
    power = np.abs(np.fft.rfft(y - y.mean())) ** 2
    freq = np.fft.rfftfreq(N)
    keep = (freq > 1 / 200) & (freq < 1 / 4)
    axes[1].semilogy(1 / freq[keep], power[keep], color="#2a6f97", lw=0.7)
    axes[1].axvline(PERIOD, color="#d1495b", ls="--", lw=0.8, label="period 24")
    axes[1].axvline(PERIOD / 2, color="#d1495b", ls=":", lw=0.8, label="period 12")
    axes[1].set(xlabel="period (steps)", ylabel="power", title="Periodogram, periods 4-200")
    axes[1].legend()
    fig.savefig(OUT / "fig_periodicity.pdf")

    runs = pd.read_csv(OUT / "ablation_runs.csv")
    base = runs[runs.seed == "-"]
    runs = runs[runs.seed != "-"]
    order = list(dict.fromkeys(runs.config))
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    for i, name in enumerate(order):
        r = runs[runs.config == name].rmse
        ax.bar(i, r.mean(), color="#43aa8b" if "future" in name else "#9db4c0")
        ax.scatter([i] * len(r), r, color="#222", s=12, zorder=3)
    for _, row in base.iterrows():
        ax.axhline(row.rmse, ls=":", color="#888", lw=0.8)
        ax.text(len(order) - 0.5, row.rmse, row.config, fontsize=7, va="bottom", ha="right")
    ax.set_xticks(range(len(order)), [n.replace(", ", "\n") for n in order], fontsize=8)
    ax.set(ylabel="validation RMSE", title="External-data ablation (bars = mean, dots = seeds 0–2)")
    fig.savefig(OUT / "fig_ablation.pdf")

    preds = {}
    for name, cfg in {"no ext": dict(BASE), "ext future": dict(BASE, ext="future")}.items():
        cfg = dict(cfg, max_epochs=4, patience=4)
        model, _, (series, marks, scaler) = run(cfg, 0, verbose=False)
        origin = N - 2 * PRED_LEN - 24 * 10
        preds[name] = scaler.inv(predict(model, series, marks, np.array([origin]), cfg["ext"], "cpu"))[0]
    t = np.arange(origin - SEQ_LEN, origin + PRED_LEN)
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    ax.plot(t, y[t], color="#222", lw=1, label="actual")
    for (name, p), c in zip(preds.items(), ["#9db4c0", "#43aa8b"]):
        ax.plot(np.arange(origin, origin + PRED_LEN), p, color=c, lw=1.6, label=name)
    ax.axvline(origin, color="#888", ls=":")
    ax.set(xlabel="time_idx - 1", ylabel="value", title="One validation window (seed 0, 4 epochs)")
    ax.legend()
    fig.savefig(OUT / "fig_val_forecast.pdf")
    print("wrote", [p.name for p in OUT.glob("fig_*.pdf")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ablate", "final", "figures"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext", choices=["none", "past", "future"], default="future")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    {"ablate": lambda: ablate(args.seeds, args.device), "final": lambda: final(args, args.device),
     "figures": figures}[args.mode]()
