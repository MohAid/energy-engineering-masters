"""
التجربة الأولى — متذبذب توافقي مخمد

مقارنة شبكتين متطابقتين في كل شيء، والفرق الوحيد بينهما أن الثانية تضع المعادلة
الحاكمة في دالة الخسارة. الأولى تتعلم من ثماني عشرة قراءة في أول 40% من الزمن،
والثانية تتعلم من القراءات نفسها بالإضافة إلى تحقيق المعادلة عند ستين نقطة على المجال كله.

المعادلة: m x'' + mu x' + k x = 0 مع x(0)=1 و x'(0)=0، وثوابتها m=1, mu=4, k=400.
التخامد ضعيف أمام القساوة، فالحركة تذبذب متناقص السعة، ولها حل تحليلي يقاس عليه.

التشغيل: python exp1_oscillator.py
المخرجات: ثلاثة أشكال وملف results.json في مجلد out_exp1
"""

# المكتبات
import os
import time
import json
from pathlib import Path
from collections.abc import Callable
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

""" MatPlotLib Configuration """
matplotlib.use("Agg")  # خلفية رسم تحفظ الأشكال في ملفات بلا فتح نافذة

""" Torch Alternatives: """
# JAX: تفاضل آلي قوي جداً ومناسب للمشتقات العالية، وسريع بفضل التجميع (jit).
#     بالمقابل أسلوبه وظيفي بحت، ومنحنى تعلمه أصعب.
# TensorFlow: استعملها Raissi-2019 وزملاؤه في الورقة الأصلية لـ PINN
#     لكن  أغلب الأبحاث الجديدة اليوم أصبحت على PyTorch أو JAX.
# DeepXDE: مكتبة عالية المستوى مبنية فوق PyTorch أو TF أو JAX.
#     بتكتب المعادلة وشروطها بسطور قليلة وهي بتبني الباقي.
#     مريحة، لكنها تخفي التفاصيل.

""" Torch Configuration """
# معايرة دقة حفظ الأرقام العشرية لأجل أي تنسور أو وزن في تورش

# مع L-BFGS ضروري أن نستعمل float64 !

# الخيارات المتاحة:
# float16: 1e-4: تسريع الشبكات الضخمة على كرت الشاشة, يعطي 4 خانات عشرية تقريباً  # noqa
# bfloat16: 1e-7: تدريب النماذج اللغوية الكبيرة, يعطي 3 إلى 4 خانات عشرية تقريباً  # noqa
# float32: 1.2e-7: الافتراضي بالتعلم العميق, يعطي 7 خانات عشرية تقريباً  # noqa
# float64: 2.2e-16: الحساب العلمي، واختيارنا, يعطي 15 إلى 16 خانة عشرية تقريباً  # noqa

# نستطيع إما تحديد سلوك المكتبة كاملة عبر:
#
# وهذا يضعف الذاكرة والسرعة ويطيل الزمن وهو أمر لا يهمنا في مثالنا هنا لأنه يحوي فقط 2209 عقدة # noqa
# كما أن بعض الأجهزة لا تدعمه،
# أو يمكن أن نحدد سلوك كائن محدد عبر استعمال:
# model = Net(LAYERS).double() # فقط الأوزان
# t = torch.tensor(data, dtype=torch.float64) # فقط التنسور

# NOTE: When we generate a torch tensor from a numpy matrix it comes with its numpy origin curacy
# np.linspace generates float64 matrix so no problem here.

# في حال استوردت دالة إلى مكان آخر فإن عملية الاستيراد ستفعل هذا الإعداد العام ! # noqa
# سيتم إعداد تورش بهذا الشكل على المستوى الأعلى وقد يؤثر على عمل المستورد ! # noqa
# حينها يجب أن تعدل السلوك ليصبح تحديد الدقة عند إنشاء الكائن وليس إعداداً عاماً ! # noqa

# قبل المعايرة:
# print(torch.tensor([1.0]).dtype)    # torch.float32  (الافتراضي)
# بعد المعايرة:
# print(torch.tensor([1.0]).dtype)    # torch.float64
torch.set_default_dtype(torch.float64)

""" Define the processing device """
DEV = torch.device("cpu")
# تحديد جهاز التدريب إما المعالج أو كرت الشاشة
# DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# نتركه على الافتراضي وهو المعالج لأن المسألة صغيرة وكلفة النقل إلى كرت الشاشة ستكون أكبر من المعالجة # noqa
# float64 أسرع على المعالج
# أيضاً بعض العمليات على كروت الشاشة غير حتمية, ذات الكود وبذات البذرة قد يعطي نتائج مختلفة في الخانات الأخيرة # noqa

""" Output Folder """
stamp = time.strftime("%Y%m%d_%H%M%S")
OUT = Path(__file__).resolve().parent / f"out_exp1_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)

""" Problem Constants """
# ثوابت المسألة: الكتلة ومعامل التخامد وقساوة النابض
# m x'' + mu x' + k x = 0
# x(0) = 1 ,  x'(0) = 0
M, MU, K = 1.0, 4.0, 400.0
DELTA = MU / (2 * M)  # معدل تناقص السعة = 2 وهو كمثال Ben Moseley المرجعي في شرح ال PINN  # noqa
OMEGA0 = np.sqrt(K / M)  # التردد الطبيعي غير المخمد = 20 وهو كمثال Ben Moseley المرجعي في شرح ال PINN  # noqa
OMEGA = np.sqrt(OMEGA0 ** 2 - DELTA ** 2)  # التردد المخمد الفعلي
T = 2 * np.pi / OMEGA  # الدور

""" Experiment (Methode) Constants """
SEED = 0  # بذرة ثابتة ليبدأ النموذجان من الأوزان نفسها
LAYERS = [1, 32, 32, 32, 1]  # بنية الشبكة: دخل، ثلاث طبقات مخفية، خرج
N_DATA, N_COLL = 18, 60  # عدد القراءات، وعدد نقاط فرض المعادلة
ADAM_STEPS, ADAM_LR = 20_000, 1e-3  # عدد الخطوات ومعدل التعلم أو حجم الخطوة
LBFGS_STEPS = 3_000


""" Validation """
# الحل التحليلي أدناه يفترض تخامداً ضعيفاً، لذا سنوقف التشغيل إذا تمت مخالفة الشرط
assert DELTA < OMEGA0, "This code is valid only for underdamped motion (DELTA < OMEGA0)"


""" The functions """


def exact(t: float | np.ndarray) -> np.float64 | np.ndarray:
    """الحل التحليلي المرجعي للمتذبذب المخمد. تستعمل هذه الدالة فقط لأجل المقارنة ولا علاقة لها بللتدريب."""  # noqa
    return np.exp(-DELTA * t) * (np.cos(OMEGA * t) + (DELTA / OMEGA) * np.sin(OMEGA * t))


class Net(nn.Module):  # الأصل الذي ترث منه أي شبكة
    """شبكة أمامية بتفعيل أملس، لأن حد الفيزياء يحتاج المشتق الثاني."""  # noqa

    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        seq = []
        # طبقة خطية بعد كل طبقة، والتفعيل بينها لا بعد الأخيرة
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))  # nn.Linear طبقة خطية
            if i < len(layers) - 2:
                seq.append(nn.Tanh())  # nn.Tanh هي دالة التفعيل
        # سلسلة طبقات
        self.net = nn.Sequential(*seq)
        # تهيئة إكسافييه تناسب التفعيل المستعمل
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # تهيئة الأوزان
                nn.init.xavier_normal_(tensor=layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """باقي المعادلة عند نقاط زمنية: مقدار مخالفة الشبكة للفيزياء."""
    # الاشتقاق هنا بالنسبة للزمن لا للأوزان
    t = t.detach().requires_grad_(True)
    x = model(t)
    # المشتق الأول ثم الثاني بالتفاضل الآلي
    dx = torch.autograd.grad(outputs=x, inputs=t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    d2x = torch.autograd.grad(outputs=dx, inputs=t, grad_outputs=torch.ones_like(dx), create_graph=True)[0]
    # التعويض في طرف المعادلة الأيسر
    return M * d2x + MU * dx + K * x


def make_loss(model: nn.Module, t_data: torch.Tensor, x_data: torch.Tensor, t_coll: torch.Tensor,
              use_physics: bool, lam: float) -> Callable[[], torch.Tensor]:
    """دالة الخسارة: حد البيانات وحده، أو معه حد الفيزياء."""
    parts = {}  # آخر قيمة لكل حد، للتسجيل فقط قد نحتاج رسمه يوماً ما !

    def loss_fn() -> torch.Tensor:
        # حد البيانات
        l_data = torch.mean((model(t_data) - x_data) ** 2)
        loss = l_data
        parts["data"] = l_data.item()
        # هذا هو الفرق الوحيد بين النموذجين
        if use_physics:
            l_phys = torch.mean(residual(model, t_coll) ** 2)
            parts["phys"] = l_phys.item()
            # الوزن لموازنة مقدارَي الحدين، لا لترجيح أحدهما
            loss = loss + lam * l_phys
        return loss
    loss_fn.parts = parts
    return loss_fn


def run(tag: str, use_physics: bool, lam: float = 1e-4, t_max_data: float = 0.4) -> dict:
    """تشغيل كامل: بناء، ثم تدريب على مرحلتين، ثم تقييم."""
    # تحقق بسيط
    assert 0.0 < t_max_data < 1.0, "observation window must lie inside (0, 1)"

    # تصفير المولدات العشوائية قبل كل تشغيل
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # للتأكد من أن الشبكات ستبدأ من بارامترات متطابقة
    # a = Net(LAYERS)
    # b = Net(LAYERS)
    # same = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    # print(same)  # True

    # القراءات داخل نافذة الرصد وحدها
    t_data_np = np.linspace(start=0.0, stop=t_max_data, num=N_DATA, endpoint=True).reshape(-1, 1)
    x_data_np = exact(t_data_np)
    t_data = torch.tensor(data=t_data_np, device=DEV)
    x_data = torch.tensor(data=x_data_np, device=DEV)
    # نقاط التجميع موزعة على المجال كله، ومنها ما لا قراءة عنده
    t_coll = torch.tensor(data=np.linspace(start=0.0, stop=1.0, num=N_COLL).reshape(-1, 1), device=DEV)

    model = Net(LAYERS).to(DEV)
    loss_fn = make_loss(model, t_data, x_data, t_coll, use_physics, lam)

    # المرحلة الأولى: استكشاف بمحسن من الرتبة الأولى
    opt = torch.optim.Adam(params=model.parameters(), lr=ADAM_LR)
    hist = []
    t0 = time.perf_counter()  # لقياس فرق الزمن الحقيقي المستهلك لتنفيذ الكود
    for step in range(ADAM_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
        # تسجيل الخسارة كل مئتي خطوة
        if step % 200 == 0:
            hist.append((step, loss.item()))
    t_adam = time.perf_counter() - t0

    # المرحلة الثانية: تحسين شبه نيوتني يقرّب الانحناء من تاريخ التدرجات  # noqa
    lbfgs = torch.optim.LBFGS(params=model.parameters(), max_iter=LBFGS_STEPS, history_size=50,
                              tolerance_grad=1e-12, tolerance_change=1e-14,
                              line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        """إعادة حساب الخسارة وتدرجها عند كل طول خطوة يجربه المحسن."""
        lbfgs.zero_grad(set_to_none=True)
        _loss = loss_fn()
        _loss.backward()
        return _loss

    # نداء واحد ينفذ التحسين كله
    t1 = time.perf_counter()
    loss_before = lbfgs.step(closure).item()
    t_lbfgs = time.perf_counter() - t1

    # فحص الأوزان
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError(f"{tag}: non-finite weights after L-BFGS")

    # عدد التكرارات المنفذة فعلاً، وبلوغ الحد يعني عدم التقارب
    n_iter = lbfgs.state_dict()["state"][0]["n_iter"]

    # الخسارة الفعلية بالأوزان النهائية
    final_loss = loss_fn().item()

    # التقييم على شبكة كثيفة مستقلة عن نقاط التدريب
    t_eval_np = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    with torch.no_grad():
        pred = model(torch.tensor(t_eval_np, device=DEV)).cpu().numpy()
    _ex = exact(t_eval_np)

    # الخطأ النسبي، لابعدي
    def rel(a, b) -> float:
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    # فصل داخل نافذة الرصد عن خارجها، وهو سؤال التجربة
    inside = t_eval_np <= t_max_data

    res = dict(tag=tag, physics=use_physics, lam=lam if use_physics else None,
               window=t_max_data, params=sum(p.numel() for p in model.parameters()),
               rel_full=rel(pred, _ex), rel_in=rel(pred[inside], _ex[inside]),
               rel_out=rel(pred[~inside], _ex[~inside]), loss_adam=loss_before, loss=final_loss,
               t_adam=t_adam, t_lbfgs=t_lbfgs, lbfgs_iters=int(n_iter))
    # مصفوفات للأشكال فقط، تحذف قبل حفظ النتائج
    res["_pred"], res["_hist"] = pred, hist
    res["_tdata"], res["_xdata"] = t_data_np, x_data_np
    # سطر تقدم لكل تشغيل
    print(f"  [{tag:22s}] full={res['rel_full']:.3e}  in={res['rel_in']:.3e}  "
          f"out={res['rel_out']:.3e}  loss={final_loss:.3e}  "
          f"adam={t_adam:.1f}s  lbfgs={t_lbfgs:.1f}s ({int(n_iter)} it)")
    return res


def line(c: str = "=") -> None:
    """سطر فاصل يبقي التقرير مقروءاً."""
    print(c * 92)


if __name__ == "__main__":
    line()
    print("EXPERIMENT 1 - DAMPED HARMONIC OSCILLATOR")
    line()
    # بيانات البيئة، لإعادة الإنتاج
    print(f"torch {torch.__version__} | device {DEV} | threads {torch.get_num_threads()} | dtype float64 | seed {SEED}")
    print(f"equation   : m x'' + mu x' + k x = 0   with m={M}, mu={MU}, k={K}")
    print(f"derived    : delta={DELTA:.4f}  omega0={OMEGA0:.4f}  omega={OMEGA:.4f}  T={T:.4f}")
    print(f"network    : {LAYERS} tanh, Xavier init")
    print(f"training   : Adam lr={ADAM_LR} x {ADAM_STEPS} steps, then L-BFGS (max {LBFGS_STEPS} it)")
    print(f"data       : {N_DATA} readings on [0, 0.4] | collocation: {N_COLL} points on [0, 1]")
    line("-")

    # نتيجة كل تشغيل
    R = {}
    # المقارنة الأساسية
    R["naive"] = run(tag="naive data only", use_physics=False)
    R["pinn"] = run(tag="PINN lam=1e-4", use_physics=True, lam=1e-4)
    # تغيير وزن الفيزياء لاختبار حساسية النتيجة له
    R["pinn_l5"] = run(tag="PINN lam=1e-5", use_physics=True, lam=1e-5)
    R["pinn_l3"] = run(tag="PINN lam=1e-3", use_physics=True, lam=1e-3)
    # توسيع نافذة الرصد لاختبار إن كان ذلك يكفي
    R["naive_w6"] = run(tag="naive window 0.6", use_physics=False, t_max_data=0.6)
    R["pinn_w6"] = run(tag="PINN window 0.6", use_physics=True, lam=1e-4, t_max_data=0.6)

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
        _lam = f"{r['lam']:.0e}" if r["lam"] else "-"
        print(f"{r['tag']:<18}{_lam:>10}{r['window']:>9}{r['rel_in']:>13.3e}{r['rel_out']:>14.3e}{r['rel_full']:>13.3e}")
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
