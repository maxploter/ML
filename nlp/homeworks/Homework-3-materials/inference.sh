#!/bin/bash -l

sentences=(
  "Barack Obama served as the President of the United States before Donald Trump."
  "Apple Inc. announced the release of the new iPhone 12 in Cupertino last October."
  "The Amazon rainforest spans across Brazil, Peru, and Colombia."
  "Leonardo DiCaprio won an Oscar for Best Actor for his role in The Revenant."
  "Microsoft acquired LinkedIn for \$26.2 billion in 2016."
  "The Mona Lisa, painted by Leonardo da Vinci, is exhibited in the Louvre Museum in Paris."
  "The Nobel Prize in Physics 2020 was awarded to Roger Penrose for his discovery in black hole formation."
  "Mount Everest, located in the Himalayas, is the highest peak on Earth."
  "Harry Potter, a series written by J.K. Rowling, became a global phenomenon."
  "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions."
)

for sentence in "${sentences[@]}"; do
    python inference.py "$sentence"
done