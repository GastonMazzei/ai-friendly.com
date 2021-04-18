#!/bin/sh

mkdir gallery >> logs.txt
echo "Generating a 3d-view of the true values and the predicted values for the testing set"
python3 scripts/view.py >> logs.txt
echo "Done! (saved in gallery)"
eog gallery/regression-testing-predictions.png >> logs.txt
echo "removing logs"
echo "Done!"
echo "ENDED"
