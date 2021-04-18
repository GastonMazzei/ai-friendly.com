#!/bin/sh

echo "...About to run all..."
bash generate_regression.sh
bash generate_gallery.sh
bash generate_classification.sh
echho "...ended running all..."
