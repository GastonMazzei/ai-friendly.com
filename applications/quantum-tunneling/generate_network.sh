#!/bin/sh


echo "Training AI-Friendly over the database!"
python3 scripts/network.py 32 20 >> logs.txt
echo "Done! results saved as 'network-results.png' in the current folder"
eog network-results.png
echo "removing logs..."
rm logs.txt
echo "Done!"
echo "ENDED"
