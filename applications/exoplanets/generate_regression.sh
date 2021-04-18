#!/bin/sh

echo "Fitting the filtered-database as a regression problem with a modified-AI-Friendly that uses a linear activation for the output layer"
python3 scripts/network_regressor.py 32 75 >> logs.txt
echo "Done! predictions saved in 'database/network-regression-prediction.csv'"
echo "removing logs..."
rm logs.txt
echo "Done!"
echo "ENDED"
