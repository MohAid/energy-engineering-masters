#!/usr/bin/env bash
# إنشاء مجلد مشروع جديد انطلاقاً من القالب.
#
# الاستخدام:
#   ./scripts/new-project.sh <course-folder> <project-slug>
# مثال:
#   ./scripts/new-project.sh pinn pinn-02-heat-conduction

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <course-folder> <project-slug>" >&2
  echo "Courses available:" >&2
  ls -1 "$(dirname "$0")/../courses" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COURSE="$1"
SLUG="$2"
DEST="$ROOT/courses/$COURSE/projects/$SLUG"

[ -d "$ROOT/courses/$COURSE" ] || { echo "No such course: $COURSE" >&2; exit 1; }
[ -e "$DEST" ] && { echo "Project already exists: $DEST" >&2; exit 1; }

cp -R "$ROOT/_templates/project-template" "$DEST"
echo "Created: courses/$COURSE/projects/$SLUG"
echo "Next: edit its README.md, then add a row to courses/$COURSE/README.md"
