#!/bin/sh
rm -r gallery
mkdir gallery
echo "1) Simulating a Sallen-Key low pass filter"
python3 scripts/generator.py 1 >> logs.txt
rm logs.txt
echo "Done! (saved in the gallery)"
echo "2) Simulating the attenuation factor as a function of frequency for different configurations"
python3 scripts/generator.py 2 >> logs.txt
rm logs.txt
echo "Done! (saved in the gallery)"
echo "3) Simulating 10k Sallen-key low pass filters with a Diode in parallel with the first resistance from source"
python3 scripts/generator.py 3 >> logs.txt
rm logs.txt
echo "Done! (saved in the database)"
echo "4) Plotting the range of the circuit elements' values used in the simulation"
python3 ../../brains/plotter.py >> logs.txt
rm logs.txt
echo "Done! (saved in the gallery)"
echo "5) Training a neural network in the classification task over the database"
python3 ../../brains/procesador.py >> logs.txt
rm logs.txt
echo "Done! (saved in the gallery)"
