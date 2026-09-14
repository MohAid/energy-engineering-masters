# -*- coding: utf-8 -*-
"""
التجربة الأولى — متذبذب توافقي مخمد

مقارنة شبكتين متطابقتين في كل شيء، والفرق الوحيد بينهما أن الثانية تضع المعادلة
الحاكمة في دالة الخسارة. الأولى تتعلم من ثماني عشرة قراءة في أول 40% من الزمن،
والثانية تتعلم من القراءات نفسها ومن المعادلة عند ستين نقطة على المجال كله.

المعادلة: m x'' + mu x' + k x = 0 مع x(0)=1 و x'(0)=0، وثوابتها m=1, mu=4, k=400.
التخامد ضعيف أمام القساوة، فالحركة تذبذب متناقص السعة، ولها حل تحليلي يقاس عليه.

التشغيل: python exp1_oscillator.py
المخرجات: ثلاثة أشكال وملف results.json في مجلد out_exp1
"""

# المكتبات
import os, time, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
# خلفية رسم تحفظ الأشكال في ملفات بلا فتح نافذة
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# الحساب بدقة مضاعفة، فالدقة المطلوبة قريبة من 1e-5
torch.set_default_dtype(torch.float64)

# الجهاز: المعالج افتراضاً
DEV = os.environ.get("PINN_DEVICE", "cpu")
if DEV == "auto":
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

# مجلد المخرجات بجانب هذا الملف
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_exp1")
os.makedirs(OUT, exist_ok=True)

# ثوابت المسألة: الكتلة ومعامل التخامد وقساوة النابض
M, MU, K = 1.0, 4.0, 400.0
# معدل تناقص السعة
DELTA = MU / (2 * M)
# التردد الطبيعي غير المخمد
OMEGA0 = np.sqrt(K / M)
# التردد المخمد الفعلي
OMEGA = np.sqrt(OMEGA0 ** 2 - DELTA ** 2)

# بذرة ثابتة ليبدأ النموذجان من الأوزان نفسها
SEED = 0
# بنية الشبكة: دخل، ثلاث طبقات مخفية، خرج
LAYERS = [1, 32, 32, 32, 1]
# عدد القراءات، وعدد نقاط فرض المعادلة
N_DATA, N_COLL = 18, 60
# مرحلتا التدريب
ADAM_STEPS, ADAM_LR = 20_000, 1e-3
LBFGS_STEPS = 3_000


def exact(t):
    """الحل التحليلي المرجعي للمتذبذب المخمد."""
    return np.exp(-DELTA * t) * (np.cos(OMEGA * t) + (DELTA / OMEGA) * np.sin(OMEGA * t))


class Net(nn.Module):
    """شبكة أمامية بتفعيل أملس، لأن حد الفيزياء يحتاج المشتق الثاني."""

    def __init__(self, layers):
        super().__init__()
        seq = []
        # طبقة خطية بعد كل طبقة، والتفعيل بينها لا بعد الأخيرة
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq.append(nn.Tanh())
        self.net = nn.Sequential(*seq)
        # تهيئة إكسافييه تناسب التفعيل المستعمل
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


def residual(model, t):
    """باقي المعادلة عند نقاط زمنية: مقدار مخالفة الشبكة للفيزياء."""
    # الاشتقاق هنا بالنسبة للزمن لا للأوزان
    t = t.requires_grad_(True)
    x = model(t)
    # المشتق الأول ثم الثاني بالتفاضل الآلي
    dx = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    d2x = torch.autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    # التعويض في طرف المعادلة الأيسر
    return M * d2x + MU * dx + K * x


def make_loss(model, t_data, x_data, t_coll, use_physics, lam):
    """دالة الخسارة: حد البيانات وحده، أو معه حد الفيزياء."""
    def loss_fn():
        # حد البيانات
        loss = torch.mean((model(t_data) - x_data) ** 2)
        # هذا هو الفرق الوحيد بين النموذجين
        if use_physics:
        # الوزن لموازنة مقدارَي الحدين، لا لترجيح أحدهما
            loss = loss + lam * torch.mean(residual(model, t_coll) ** 2)
        return loss
    return loss_fn


def run(tag, use_physics, lam=1e-4, t_max_data=0.4):
    """تشغيل كامل: بناء، ثم تدريب على مرحلتين، ثم تقييم."""
    # تصفير المولدات العشوائية قبل كل تشغيل
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # القراءات داخل نافذة الرصد وحدها
    t_data_np = np.linspace(0.0, t_max_data, N_DATA).reshape(-1, 1)
    t_data = torch.tensor(t_data_np, device=DEV)
    x_data = torch.tensor(exact(t_data_np), device=DEV)
    # نقاط التجميع موزعة على المجال كله، ومنها ما لا قراءة عنده
    t_coll = torch.tensor(np.linspace(0.0, 1.0, N_COLL).reshape(-1, 1), device=DEV)

    model = Net(LAYERS).to(DEV)
    loss_fn = make_loss(model, t_data, x_data, t_coll, use_physics, lam)

    # المرحلة الأولى: استكشاف بمحسن من الرتبة الأولى
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    hist = []
    t0 = time.perf_counter()
    for step in range(ADAM_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
        # تسجيل الخسارة كل مئتي خطوة
        if step % 200 == 0:
            hist.append((step, loss.item()))
    t_adam = time.perf_counter() - t0

    # المرحلة الثانية: تحسين شبه نيوتني يقرّب الانحناء من تاريخ التدرجات
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_STEPS, history_size=50,
                              tolerance_grad=1e-12, tolerance_change=1e-14,
                              line_search_fn="strong_wolfe")

    def closure():
        """إعادة حساب الخسارة وتدرجها عند كل طول خطوة يجربه المحسن."""
        lbfgs.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return loss

    t1 = time.perf_counter()
    # نداء واحد ينفذ التحسين كله
    final_loss = lbfgs.step(closure).item()
    t_lbfgs = time.perf_counter() - t1
    # عدد التكرارات المنفذة فعلاً، وبلوغ الحد يعني عدم التقارب
    n_iter = lbfgs.state_dict()["state"][0]["n_iter"]

    # التقييم على شبكة كثيفة مستقلة عن نقاط التدريب
    t_eval_np = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    with torch.no_grad():
        pred = model(torch.tensor(t_eval_np, device=DEV)).cpu().numpy()
    ex = exact(t_eval_np)
    # الخطأ النسبي، لابعدي
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    # فصل داخل نافذة الرصد عن خارجها، وهو سؤال التجربة
    inside = t_eval_np <= t_max_data

    res = dict(tag=tag, physics=use_physics, lam=lam if use_physics else None,
               window=t_max_data, params=sum(p.numel() for p in model.parameters()),
               rel_full=rel(pred, ex), rel_in=rel(pred[inside], ex[inside]),
               rel_out=rel(pred[~inside], ex[~inside]), loss=final_loss,
               t_adam=t_adam, t_lbfgs=t_lbfgs, lbfgs_iters=int(n_iter))
    # مصفوفات للأشكال فقط، تحذف قبل حفظ النتائج
    res["_pred"], res["_hist"] = pred, hist
    res["_tdata"], res["_xdata"] = t_data_np, exact(t_data_np)
    # سطر تقدم لكل تشغيل
    print(f"  [{tag:22s}] full={res['rel_full']:.3e}  in={res['rel_in']:.3e}  "
          f"out={res['rel_out']:.3e}  loss={final_loss:.3e}  "
          f"adam={t_adam:.1f}s  lbfgs={t_lbfgs:.1f}s ({int(n_iter)} it)")
    return res


def line(c="="):
    """سطر فاصل يبقي التقرير مقروءاً."""
    print(c * 92)


if __name__ == "__main__":
    line()
    print("EXPERIMENT 1 - DAMPED HARMONIC OSCILLATOR")
    line()
    # بيانات البيئة، لإعادة الإنتاج
    print(f"torch {torch.__version__} | device {DEV} | threads {torch.get_num_threads()} | dtype float64 | seed {SEED}")
    print(f"equation   : m x'' + mu x' + k x = 0   with m={M}, mu={MU}, k={K}")
    print(f"derived    : delta={DELTA:.4f}  omega0={OMEGA0:.4f}  omega={OMEGA:.4f}  period={2*np.pi/OMEGA:.4f}")
    print(f"network    : {LAYERS} tanh, Xavier init")
    print(f"training   : Adam lr={ADAM_LR} x {ADAM_STEPS} steps, then L-BFGS (max {LBFGS_STEPS} it)")
    print(f"data       : {N_DATA} readings on [0, 0.4] | collocation: {N_COLL} points on [0, 1]")
    line("-")

    # نتيجة كل تشغيل
    R = {}
    # المقارنة الأساسية
    R["naive"] = run("naive data only", False)
    R["pinn"] = run("PINN lam=1e-4", True, lam=1e-4)
    # تغيير وزن الفيزياء لاختبار حساسية النتيجة له
    R["pinn_l5"] = run("PINN lam=1e-5", True, lam=1e-5)
    R["pinn_l3"] = run("PINN lam=1e-3", True, lam=1e-3)
    # توسيع نافذة الرصد لاختبار هل تكفي زيادة البيانات وحدها
    R["naive_w6"] = run("naive window 0.6", False, t_max_data=0.6)
    R["pinn_w6"] = run("PINN window 0.6", True, lam=1e-4, t_max_data=0.6)

    line()
    print("TABLE 1 - SETUP")
    line("-")
    print(f"{'item':<28}{'value'}")
    # الإعداد كاملاً في جدول واحد
    for k, v in [("equation", "m x'' + mu x' + k x = 0"), ("constants", f"m={M}, mu={MU}, k={K}"),
                 ("initial conditions", "x(0)=1, x'(0)=0"), ("network", f"{LAYERS} tanh"),
                 ("trainable parameters", R["naive"]["params"]), ("readings", f"{N_DATA} on [0, 0.4]"),
                 ("collocation points", f"{N_COLL} on [0, 1]"), ("physics weight", "1e-4 (also 1e-5, 1e-3)"),
                 ("optimiser", f"Adam {ADAM_LR} x {ADAM_STEPS}, then L-BFGS"),
                 ("evaluation grid", "400 points on [0, 1]"), ("seed", SEED),
                 ("environment", f"torch {torch.__version__}, float64, {torch.get_num_threads()} threads")]:
        print(f"{k:<28}{v}")

    line()
    print("TABLE 2 - MAIN COMPARISON")
    line("-")
    hdr = f"{'model':<18}{'err inside':>13}{'err outside':>14}{'err full':>13}{'adam s':>9}{'lbfgs s':>9}{'lbfgs it':>10}{'final loss':>13}"
    print(hdr)
    for k in ["naive", "pinn"]:
        r = R[k]
        print(f"{r['tag']:<18}{r['rel_in']:>13.3e}{r['rel_out']:>14.3e}{r['rel_full']:>13.3e}"
              f"{r['t_adam']:>9.1f}{r['t_lbfgs']:>9.1f}{r['lbfgs_iters']:>10d}{r['loss']:>13.3e}")

    line()
    print("TABLE 3 - VARIANTS")
    line("-")
    print(f"{'run':<18}{'lambda':>10}{'window':>9}{'err inside':>13}{'err outside':>14}{'err full':>13}")
    for k in ["pinn_l5", "pinn", "pinn_l3", "naive_w6", "pinn_w6"]:
        r = R[k]
        # التشغيلات بلا فيزياء لا وزن لها
        lam = f"{r['lam']:.0e}" if r["lam"] else "-"
        print(f"{r['tag']:<18}{lam:>10}{r['window']:>9}{r['rel_in']:>13.3e}{r['rel_out']:>14.3e}{r['rel_full']:>13.3e}")
    line()

    # الأشكال
    t_eval = np.linspace(0, 1, 400).reshape(-1, 1)
    ex = exact(t_eval)

    # الشكل 1: النتيجة الأساسية
    plt.figure(figsize=(9, 4.6))
    plt.plot(t_eval, ex, "k-", lw=2, label="Exact")
    plt.plot(t_eval, R["naive"]["_pred"], "r--", lw=2, label="NN (data only)")
    plt.plot(t_eval, R["pinn"]["_pred"], "b-.", lw=2, label="PINN")
    plt.scatter(R["naive"]["_tdata"], R["naive"]["_xdata"], s=30, c="g", zorder=5, label="Training data (t <= 0.4)")
    plt.axvspan(0, 0.4, alpha=0.08, color="g")
    plt.xlabel("t"); plt.ylabel("x(t)"); plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1_naive_vs_pinn.png"), dpi=200); plt.close()

    # الشكل 2: هبوط الخسارة على محور لوغاريتمي، والمنحنيان غير قابلين للمقارنة
    plt.figure(figsize=(7, 4))
    for k, style in [("naive", "r--"), ("pinn", "b-")]:
        h = np.array(R[k]["_hist"])
        plt.semilogy(h[:, 0], h[:, 1], style, label=R[k]["tag"])
    plt.xlabel("iteration"); plt.ylabel("training loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_losses.png"), dpi=200); plt.close()

    # الشكل 3: المتغيرات مقابل الحل التحليلي
    plt.figure(figsize=(9, 4.6))
    plt.plot(t_eval, ex, "k-", lw=2, label="Exact")
    plt.plot(t_eval, R["pinn_l3"]["_pred"], "-.", lw=1.8, label="PINN, lam=1e-3")
    plt.plot(t_eval, R["pinn_l5"]["_pred"], "-", lw=1.2, alpha=.8, label="PINN, lam=1e-5")
    plt.plot(t_eval, R["naive_w6"]["_pred"], "--", lw=1.8, label="NN data only, window 0.6")
    plt.plot(t_eval, R["pinn_w6"]["_pred"], ":", lw=2.2, label="PINN, window 0.6")
    plt.xlabel("t"); plt.ylabel("x(t)"); plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig3_variants.png"), dpi=200); plt.close()

    # حفظ الأرقام وحدها في ملف النتائج
    slim = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in R.items()}
    json.dump(slim, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"figures and results.json saved in: {OUT}")
    line()
