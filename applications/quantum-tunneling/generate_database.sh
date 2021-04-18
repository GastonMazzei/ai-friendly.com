#!/bin/sh

echo "Generating database... (may take ~10 minutes; alternatively a .zip'ed one is provided at the official repository GastonMazzei/AIFriendly-quantum-tunnelling. It is in the 'database' directory.)"
mkdir database >> logs.txt
python3 scripts/main.py >> logs.txt
echo "Done!"
echo "removing logs..."
rm logs.txt
echo "Done!"
echo "ENDED"
