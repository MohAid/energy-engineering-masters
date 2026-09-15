# Setup

<div dir="rtl">

كيفية تشغيل أي مشروع في هذا المستودع. أما قواعد التسمية والكتابة فموضعها [WORKFLOW.md](WORKFLOW.md).

## المتطلبات

Git، و Python 3.10 أو أحدث. بعض المشاريع لاحقاً قد تحتاج MATLAB أو OpenFOAM، ويذكر ذلك ملف README الخاص بها.

## الاستنساخ

</div>

```bash
git clone https://github.com/MohAid/energy-engineering-masters.git
cd energy-engineering-masters
```

<div dir="rtl">

## تشغيل مشروع

لكل مشروع ملف `requirements.txt` خاص به. اعمل داخل مجلد المشروع لا في جذر المستودع:

</div>

```bash
cd projects/pinn/pinn-01-oscillator-foundations

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python src/exp1_oscillator.py
```

<div dir="rtl">

المخرجات تكتب في مجلد ينشئه البرنامج وهو مستبعد من git، لأن الأشكال والأرقام النهائية مرفوعة أصلاً في `results/`.

## التشغيل على المتصفح

المشروع الذي يحوي دفتر `.ipynb` يعمل على Google Colab بلا تثبيت شيء، من زر **Open in Colab** في ملف README الخاص به.

## مشروع جديد

</div>

```bash
./scripts/new-project.sh 01-advanced-fluid-mechanics boundary-layer
```

<div dir="rtl">

ينسخ السكربت القالب من `_templates/project-template/` إلى مجلد مشاريع المقرر. أما المشروع المستقل عن المقررات فيوضع تحت `projects/` بنسخ القالب يدوياً:

</div>

```bash
cp -r _templates/project-template projects/pinn/pinn-02-heat-conduction
```

<div dir="rtl">

بعدها في الحالتين: املأ README المشروع، وأضف سطراً في جدول القسم، وسطراً في جدول المستودع الرئيسي.

## الهوية عند الرفع

تُضبط مرة واحدة على كل جهاز:

</div>

```bash
git config --global user.name "Mohamad Aid"
git config --global user.email "eng.m.n.aid@gmail.com"
```

<div dir="rtl">

وحين يطلب GitHub كلمة مرور فهو يريد personal access token لا كلمة مرور الحساب.

## صفحة العرض

ملف `docs/index.html` صفحة ساكنة بلا خطوة بناء، منشورة عبر GitHub Pages على:
<https://mohaid.github.io/energy-engineering-masters/>

وتتحدث تلقائياً عند كل رفع إلى الفرع `main`.

</div>
