# Energy Engineering — M.Sc.

<div dir="rtl">

### هندسة الطاقة — أعمال الماجستير

مستودع أجمع فيه ما أكتبه من كود خلال ماجستير هندسة الطاقة. ما هو موجود هنا منجز فعلاً، ولا يُنشأ مجلد قبل أن يصير فيه عمل يستحق الرفع.

والعمل موزع على قسمين: `courses` لما يخص مقرراً بعينه، و`projects` لما يخص الماجستير عموماً لا مقرراً واحداً، وفيه يوضع مشروع التخرج حين يأتي دوره.

صفحة العرض: <https://mohaid.github.io/energy-engineering-masters/>

## المشاريع

| المشروع | الموضع | الحالة |
|---|---|:---:|
| [fanno-flow](courses/01-advanced-fluid-mechanics/projects/fanno-flow) | مقرر [ميكانيك الموائع المتقدم](courses/01-advanced-fluid-mechanics) | منجز |
| [pinn-01-oscillator-foundations](projects/pinn/pinn-01-oscillator-foundations) | سلسلة [الشبكات المستنيرة بالفيزياء](projects/pinn) | منجز |

## البنية

</div>

```
courses/                  ما يخص مقرراً بعينه
  NN-course-name/
    README.md
    projects/
      project-name/
        README.md         المسألة والطريقة والنتائج
        src/              الكود
        results/          الأشكال والمخرجات
projects/                 ما يخص الماجستير لا مقرراً بعينه
  series-or-project-name/
    README.md
    project-name/         بالبنية نفسها أعلاه
docs/                     صفحة عرض بسيطة (GitHub Pages)
_templates/               قالب مشروع جديد
scripts/                  سكربت إنشاء مشروع مقرر من القالب
```

<div dir="rtl">

## مشروع جديد

</div>

```bash
./scripts/new-project.sh 02-refrigeration-and-air-conditioning absorption-chiller
```

<div dir="rtl">

انظر [WORKFLOW.md](WORKFLOW.md) لقواعد التسمية والالتزام، و[SETUP.md](SETUP.md) لتشغيل أي مشروع هنا.

## الترخيص

الكود تحت رخصة [MIT](LICENSE). المواد المكتوبة متاحة للاطلاع الأكاديمي مع الاستشهاد.

## المؤلف

م. محمد عيد — طالب ماجستير، هندسة الطاقة، كلية الهندسة الميكانيكية، جامعة حلب.

</div>
