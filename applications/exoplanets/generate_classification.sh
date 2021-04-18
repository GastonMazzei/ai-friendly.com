#!/bin/sh

mkdir gallery
echo "AI-Friendly is about to learn which planets have orbital periods longer than a human life (80 years)"
python3 scripts/aifriendly.py 32 50 >> logs.txt
echo "Done! results saved in 'gallery/results.png'"
eog gallery/results.png >> logs.txt
echo "removing logs"
rm logs.txt
echo "Done!"
echo "ENDED"


