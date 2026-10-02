import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

OUT = "/home/claude/fig"; os.makedirs(OUT, exist_ok=True)
SEED, EPOCHS, BATCH = 42, 100, 16
CLASSES = ["Setosa", "Versicolor", "Virginica"]

# ---------- data ----------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
X, y = iris.data.values.astype("float32"), iris.target.values.astype("int64")
X_trv, X_test, y_trv, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_trv, y_trv, test_size=0.2, random_state=SEED, stratify=y_trv)
sc = StandardScaler().fit(X_tr)
X_tr, X_val, X_test = [sc.transform(a).astype("float32") for a in (X_tr, X_val, X_test)]
print("shapes", X_tr.shape, X_val.shape, X_test.shape)

# ---------- network (4-16-8-3, ReLU, ReLU, Softmax) ----------
def init_params(seed):
    rng = np.random.default_rng(seed)
    sizes = [4, 16, 8, 3]; P = []
    for i in range(3):
        lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))           # Glorot uniform (Keras default)
        P += [rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])), np.zeros(sizes[i + 1])]
    return P

def forward(P, X):
    z1 = X @ P[0] + P[1]; a1 = np.maximum(z1, 0)
    z2 = a1 @ P[2] + P[3]; a2 = np.maximum(z2, 0)
    z3 = a2 @ P[4] + P[5]; z3 = z3 - z3.max(1, keepdims=True)
    e = np.exp(z3); p = e / e.sum(1, keepdims=True)
    return z1, a1, z2, a2, p

def loss_acc(P, X, y):
    p = forward(P, X)[-1]; p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.log(p[np.arange(len(y)), y]).mean(), (p.argmax(1) == y).mean()

def grads(P, X, y):
    z1, a1, z2, a2, p = forward(P, X); B = len(y)
    d3 = p.copy(); d3[np.arange(B), y] -= 1; d3 /= B
    gW3, gb3 = a2.T @ d3, d3.sum(0)
    d2 = (d3 @ P[4].T) * (z2 > 0); gW2, gb2 = a1.T @ d2, d2.sum(0)
    d1 = (d2 @ P[2].T) * (z1 > 0); gW1, gb1 = X.T @ d1, d1.sum(0)
    return [gW1, gb1, gW2, gb2, gW3, gb3]

def train(opt, lr, seed=SEED, epochs=EPOCHS, batch=BATCH, data_seed=SEED):
    P = init_params(data_seed)             # identical initial weights for every optimiser
    rng = np.random.default_rng(seed)      # controls shuffling order
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]; t = 0
    H = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    n = len(y_tr)
    for ep in range(epochs):
        idx = rng.permutation(n); ls = 0.0; cs = 0
        for s in range(0, n, batch):
            b = idx[s:s + batch]; xb, yb = X_tr[b], y_tr[b]
            l, a = loss_acc(P, xb, yb); ls += l * len(b); cs += a * len(b)   # running average, as Keras reports
            G = grads(P, xb, yb); t += 1
            if opt == "sgd":
                for i in range(6): P[i] -= lr * G[i]
            else:  # Adam, Keras formulation (beta1=.9, beta2=.999, eps=1e-7)
                b1, b2, eps = 0.9, 0.999, 1e-7
                lr_t = lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                for i in range(6):
                    m[i] = b1 * m[i] + (1 - b1) * G[i]
                    v[i] = b2 * v[i] + (1 - b2) * G[i] ** 2
                    P[i] -= lr_t * m[i] / (np.sqrt(v[i]) + eps)
        vl, va = loss_acc(P, X_val, y_val)
        H["loss"].append(ls / n); H["accuracy"].append(cs / n)
        H["val_loss"].append(vl); H["val_accuracy"].append(va)
    return P, H

def first_epoch(series, cond):
    for i, s in enumerate(series):
        if cond(s): return i + 1
    return None

res = {}
models = {}
for name, opt, lr in [("SGD", "sgd", 0.01), ("Adam", "adam", 0.001)]:
    P, H = train(opt, lr); models[name] = (P, H)
    tl, ta = loss_acc(P, X_test, y_test)
    pred = forward(P, X_test)[-1].argmax(1)
    cm = confusion_matrix(y_test, pred)
    rep = classification_report(y_test, pred, target_names=CLASSES, output_dict=True, digits=4)
    rep_txt = classification_report(y_test, pred, target_names=CLASSES, digits=4)
    res[name] = dict(lr=lr, history=H, test_loss=float(tl), test_acc=float(ta), cm=cm.tolist(), report=rep,
                     report_txt=rep_txt, pred=pred.tolist(),
                     ep_trainloss_05=first_epoch(H["loss"], lambda x: x <= 0.5),
                     ep_trainloss_02=first_epoch(H["loss"], lambda x: x <= 0.2),
                     ep_valacc_90=first_epoch(H["val_accuracy"], lambda x: x >= 0.9),
                     ep_valloss_min=int(np.argmin(H["val_loss"])) + 1, valloss_min=float(min(H["val_loss"])))
    print(f"\n=== {name} (lr={lr}) ===")
    for k in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        print(k, "ep1=%.4f ep10=%.4f ep50=%.4f ep100=%.4f" % (H[k][0], H[k][9], H[k][49], H[k][-1]))
    print("test loss %.4f  test acc %.4f" % (tl, ta)); print(cm); print(rep_txt)
    print({k: res[name][k] for k in ["ep_trainloss_05", "ep_trainloss_02", "ep_valacc_90", "ep_valloss_min", "valloss_min"]})
res["y_test"] = y_test.tolist()

# parameter count
P0 = init_params(SEED); res["n_params"] = int(sum(p.size for p in P0))
print("params", res["n_params"])

# ---------- learning-rate experiment (SGD) ----------
lr_res = {}
for lr in [0.001, 0.01, 0.1]:
    P, H = train("sgd", lr)
    tl, ta = loss_acc(P, X_test, y_test)
    lr_res[str(lr)] = dict(history=H, final_loss=H["loss"][-1], final_val_loss=H["val_loss"][-1],
                           final_acc=H["accuracy"][-1], final_val_acc=H["val_accuracy"][-1],
                           test_loss=float(tl), test_acc=float(ta),
                           ep_trainloss_05=first_epoch(H["loss"], lambda x: x <= 0.5),
                           ep_trainloss_02=first_epoch(H["loss"], lambda x: x <= 0.2),
                           max_loss_increase=float(max(np.diff(H["loss"])) if len(H["loss"]) > 1 else 0),
                           n_up=int(sum(np.diff(H["val_loss"]) > 0)))
    print("LR", lr, {k: v for k, v in lr_res[str(lr)].items() if k != "history"})
res["lr_exp"] = lr_res

# ---------- supplementary: 10 repeats (different shuffling seeds; same split and same init) ----------
rep_out = {}
for name, opt, lr in [("SGD", "sgd", 0.01), ("Adam", "adam", 0.001)]:
    accs, losses, e02 = [], [], []
    for s in range(10):
        # vary both init and shuffling
        Pw = None
        P_, H_ = train(opt, lr, seed=1000 + s, data_seed=2000 + s)
        tl, ta = loss_acc(P_, X_test, y_test)
        accs.append(ta); losses.append(tl); e02.append(first_epoch(H_["loss"], lambda x: x <= 0.2))
    rep_out[name] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)), acc_min=float(min(accs)),
                         acc_max=float(max(accs)), loss_mean=float(np.mean(losses)), loss_std=float(np.std(losses)),
                         ep02=e02)
    print("repeat", name, rep_out[name])
res["repeats"] = rep_out

json.dump(res, open("/home/claude/results.json", "w"), indent=1, default=str)

# ---------- figures ----------
sns.set_theme(style="whitegrid")
colors = {"SGD": "#d1495b", "Adam": "#2e6f95"}
ep = np.arange(1, EPOCHS + 1)
df["species"] = df["target"].map(dict(enumerate(CLASSES)))

fig, ax = plt.subplots(figsize=(5, 3.4))
cnt = df["species"].value_counts().reindex(CLASSES)
ax.bar(CLASSES, cnt.values, color=["#4c9f70", "#e0a458", "#6a5acd"])
for i, c in enumerate(cnt.values): ax.text(i, c + 1, str(c), ha="center")
ax.set_ylabel("Number of samples"); ax.set_title("Class distribution of the Iris dataset"); ax.set_ylim(0, 60)
fig.tight_layout(); fig.savefig(f"{OUT}/class_distribution.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for sp, c in zip(CLASSES, ["#4c9f70", "#e0a458", "#6a5acd"]):
    d = df[df.species == sp]
    axs[0].scatter(d["petal length (cm)"], d["petal width (cm)"], label=sp, color=c, edgecolor="k", s=30)
    axs[1].scatter(d["sepal length (cm)"], d["sepal width (cm)"], label=sp, color=c, edgecolor="k", s=30)
axs[0].set_xlabel("Petal length (cm)"); axs[0].set_ylabel("Petal width (cm)"); axs[0].set_title("Petal length vs petal width")
axs[1].set_xlabel("Sepal length (cm)"); axs[1].set_ylabel("Sepal width (cm)"); axs[1].set_title("Sepal length vs sepal width")
axs[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/scatter.png", dpi=200); plt.close(fig)

pp = sns.pairplot(df.drop(columns="target").rename(columns=lambda c: c.replace(" (cm)", "")), hue="species",
                  palette=["#4c9f70", "#e0a458", "#6a5acd"], height=1.7, corner=False)
pp.savefig(f"{OUT}/pairplot.png", dpi=150); plt.close("all")

def curve(key, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for n in ["SGD", "Adam"]:
        ax.plot(ep, res[n]["history"][key], label=f"{n} (lr={res[n]['lr']})", color=colors[n], lw=2)
    ax.set_xlabel("Epochs"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}.png", dpi=200); plt.close(fig)
curve("loss", "Training loss: SGD vs Adam", "Loss", "train_loss")
curve("val_loss", "Validation loss: SGD vs Adam", "Loss", "val_loss")
curve("accuracy", "Training accuracy: SGD vs Adam", "Accuracy", "train_acc")
curve("val_accuracy", "Validation accuracy: SGD vs Adam", "Accuracy", "val_acc")

# per-optimizer 2-panel
for n in ["SGD", "Adam"]:
    H = res[n]["history"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(ep, H["loss"], label="Training", color=colors[n], lw=2); axs[0].plot(ep, H["val_loss"], "--", label="Validation", color="#555", lw=2)
    axs[0].set_title(f"{n}: loss"); axs[0].set_xlabel("Epochs"); axs[0].set_ylabel("Loss"); axs[0].legend()
    axs[1].plot(ep, H["accuracy"], label="Training", color=colors[n], lw=2); axs[1].plot(ep, H["val_accuracy"], "--", label="Validation", color="#555", lw=2)
    axs[1].set_title(f"{n}: accuracy"); axs[1].set_xlabel("Epochs"); axs[1].set_ylabel("Accuracy"); axs[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{n.lower()}_history.png", dpi=200); plt.close(fig)

for n in ["SGD", "Adam"]:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay(np.array(res[n]["cm"]), display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix: {n}"); ax.grid(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/cm_{n.lower()}.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
for lr, c in zip(["0.001", "0.01", "0.1"], ["#8d99ae", "#d1495b", "#2b2d42"]):
    axs[0].plot(ep, lr_res[lr]["history"]["loss"], label=f"lr={lr}", color=c, lw=2)
    axs[1].plot(ep, lr_res[lr]["history"]["val_loss"], label=f"lr={lr}", color=c, lw=2)
axs[0].set_title("SGD training loss for different learning rates"); axs[1].set_title("SGD validation loss for different learning rates")
for a in axs: a.set_xlabel("Epochs"); a.set_ylabel("Loss"); a.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/lr_experiment.png", dpi=200); plt.close(fig)

# architecture diagram
fig, ax = plt.subplots(figsize=(9, 4.4)); ax.axis("off")
layers = [(4, "Input layer\n4 features", "#cfe8ef"), (16, "Hidden layer 1\n16 neurons, ReLU", "#fde2b3"),
          (8, "Hidden layer 2\n8 neurons, ReLU", "#fde2b3"), (3, "Output layer\n3 neurons, Softmax", "#d4ecd0")]
xs = [0.1, 0.37, 0.64, 0.9]; pos = []
for (n, lab, col), x in zip(layers, xs):
    shown = min(n, 8) if n == 16 else n
    ys = np.linspace(0.78, 0.22, shown) if shown > 1 else [0.5]
    pos.append([(x, y_) for y_ in ys])
for a, b in zip(pos[:-1], pos[1:]):
    for p1 in a:
        for p2 in b: ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#bbb", lw=0.4, zorder=1)
for (n, lab, col), x, pts in zip(layers, xs, pos):
    for (px, py) in pts: ax.scatter(px, py, s=330, color=col, edgecolor="#333", zorder=3)
    ax.text(x, 0.93, lab, ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(0.37, 0.11, "(8 of 16 neurons drawn)", ha="center", fontsize=8, style="italic")
for x, t in zip(xs, ["Input\nfeatures", "", "", "Predicted\nclass"]): pass
ax.text(0.1, 0.05, "Sepal L, Sepal W,\nPetal L, Petal W", ha="center", fontsize=8)
ax.text(0.9, 0.05, "Setosa / Versicolor /\nVirginica", ha="center", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f"{OUT}/architecture.png", dpi=200); plt.close(fig)
print("done")
