#!/bin/sh
rm -r gallery
rm database/database.csv
mkdir gallery
mkdir database
echo "1) Simulating an amplifier circuit"
python3 scripts/generator.py 1 >> logs.txt
rm logs.txt
echo "Done! (saved in the gallery)"
echo "2) Simulating the amplification factor as a function of frequency for different configurations"
python3 scripts/generator.py 2 >> logs.txt
rm logs.txt
echo "Done! (saved in the gallery)"
echo "3) Simulating 12k amplifier circuits with a Diode connecting the (only) transistor's emmisor and collector (please note that the execution time will be comparatively large (e.g. 3 mins); the most probable cause is the circuit having a transistor)"
for i in {1..6}
do
  python3 scripts/generator.py 3 >> logs.txt
done
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
