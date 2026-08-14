#!/bin/bash
cd /Users/monika/GraphNeuralNetwork_Thesis/praca-dyplomowa-latex
latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode main.tex
echo "--- ERRORS FOUND ---"
grep -n -i "error" main.blg main.log 2>/dev/null