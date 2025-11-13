#!/bin/bash
# ======================================
# Script : sync_before_push.sh
# Objectif : Convertir tous les notebooks .ipynb en .py avant le push
# ======================================

echo "🔄 Conversion de tous les notebooks (.ipynb) vers scripts (.py)..."

# Parcourt tous les notebooks dans les dossiers mini-project-*/
for notebook in mini-project-*/*.ipynb mini-project-/*.ipynb; do
    if [ -f "$notebook" ]; then
        py_file="${notebook%.ipynb}.py"
        echo "→ Conversion : $notebook → $py_file"
        jupytext --to py:percent "$notebook"
    fi
done

echo "✅ Tous les notebooks ont été convertis en fichiers Python."
echo "👉 Vous pouvez maintenant :"
echo "   git add ."
echo "   git commit -m \"feat: [votre_nom] maj des notebooks\""
echo "   git push"
