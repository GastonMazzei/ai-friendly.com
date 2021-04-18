#!/bin/sh

rm -r database
echo "creating a directory to place the database converted from LHCO format to CSV"
mkdir database
echo "processing the LHCO Higgs' Signal into CSV"
python3 scripts/lhco2csv.py signal >> logs.txt
echo "Done!"
echo "processing the LHCO Background Signal into CSV"
python3 scripts/lhco2csv.py background >> logs.txt
echo "Done!"
echo "processing the CSV Background and Signal into training, validation & testing databases"
python3 scripts/full_database.py >> logs.txt
echo "Done!"
echo "Processing the training, validation and testing databases with the LHCO format into kinematic databases: only the particle's dynamic quantities will appear. There will be expressed in two ways: in spherical coordinates (as is customary in HEP; pseudorapidity, azimuth, transverse momentum and invariant mass (as in LHCO) and in cartesian coordinates (P0,Px,Py and Pz)"
python3 scripts/kinematic_databases.py >> logs.txt
echo "Done!"
echo "removing logs"
rm logs.txt
echo "Done!"
echo "ENDED"
