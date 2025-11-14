#!/bin/bash
# ======================================
# Script : sync_after_pull.sh
# Objectif : Convertir tous les scripts .py en .ipynb après le pull
# ======================================

echo "🔄 Conversion de tous les scripts (.py) vers notebooks (.ipynb)..."

# Parcourt tous les fichiers Python dans les dossiers mini-project-*/
for py_file in mini-project-*/*.py mini-project-/*.py; do
    if [ -f "$py_file" ]; then
        ipynb_file="${py_file%.py}.ipynb"
        echo "→ Conversion : $py_file → $ipynb_file"
        jupytext --to ipynb "$py_file"
    fi
done

echo "✅ Tous les fichiers Python ont été convertis en notebooks."
