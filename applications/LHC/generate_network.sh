#!/bin/bash



mkdir results
# 
if false
  then
    #                --MANUAL OPTIONS--
    #   1: training concentration of the signal (default=50)
    #   2: validation & testing concentration of the signal (default=50)
    #   3: neurons per layer (default: 32)
    #   4: epochs (default: 120; there is an EarlyStop tho...)
    #   5: batch  size (default: 32)
    #   6: type (options are: 'o' for original, 'spherical' or 'cartesian')
    #   7: depth (0 for 2 hidden layers and 1 for 3 hidden layers)
    echo $1 $2 $3 $4 $5 $6 $7
    python3 testing-script.py "$1" "$2" "$3" "$4" "$5" "$6" "$7"
  else
    echo "training AI-Friendly over the full lhco database"
    python3 scripts/aifriendly.py 50 50 32 120 32 0 0 >> logs.txt
    echo "Done!\ntraining AI-Friendly over the lhco database with only kinematic features in the original coordinates (spherical)"
    python3 scripts/aifriendly.py 50 50 32 120 32 spherical 0 >> logs.txt
    echo "Done!\ntraining AI-Friendly over the lhco database with only kinematic features in cartesian coordinates"
    python3 scripts/aifriendly.py 50 50 32 120 32 cartesian 0 >> logs.txt
    echo "Done!\ntraining AI-Friendly over the lhco database with only kinematic features in cartesian coordinates AND with 2 extra layers that could allow it to learn to internally transform from cartesian to spherical coordinates"
    python3 scripts/aifriendly.py 50 50 32 120 32 cartesian 1 >> logs.txt
    echo "Done!\nALL RESULTS SAVED IN THE DIRECTORY './results'"
fi
echo "removing logs..."
rm logs.txt
echo "Done!"
echo "ENDED"
