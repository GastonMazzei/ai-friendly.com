#!/bin/sh

echo "About to generate the database for the grou-theory classification problem shown in the paper 'Machine Learning Lie Structures & Applications to Physics'"
mkdir database
python3 scripts/make_database.py >> logs.txt
echo "Done!"
echo "About to train AI-Friendly in the classification task defined by 'scripts/make_database.py's threshold..."
mkdir gallery
python3 scripts/aifriendly.py 32 100 >> logs.txt
echo "Done!"
echo "Removing logs..."
rm logs.txt
echo "Done!"
echo "Results saved in 'gallery/results.png'"
echo "ENDED"
