#!/bin/sh

mkdir database
echo "generating 2-dimensional second-order ordinary differential equations and calculating their fixed points"
echo "WARNING: it's a cpu-intensive script. Running in Google Colab the script 'scripts/for_colab.ipynb' is highly recommended"
python3 scripts/generate_equations_and_solutions.py >> logs.txt
echo "Done! (saved in 'database/equations_and_solutions.pkl')"
echo "generating a database of 2-dimensional second-order ODE's where they are marked with '1' if it exists a stationary point which is forbidden at T=inf (unstable equilibrium) or else they are marked with '0'"
python3 scripts/generate_database.py >> logs.txt
echo "Done! (saved in 'database/database.csv')"
echo "removing logs..."
rm logs.txt
rm time-logs.txt
echo "Done!"
echo "ENDED"
