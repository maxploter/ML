# Bishop — Pattern Recognition and Machine Learning (2006)

Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
Microsoft Research hosts a free PDF:
https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
Don't commit the PDF to this repo. Link to it instead.

## Exercises

Each exercise is marked with a difficulty: (\*) takes a few minutes, (\*\*) up to about an hour,
and (\*\*\*) is harder. Exercises marked **www** have worked solutions in Bishop's official
solutions manual. Look at those only after you have made a serious attempt yourself.

## Layout

One folder per chapter, `chNN_<topic>/`. It is created when the chapter is reached, not ahead of
time.

- `exercises.md` holds one `## Exercise N.k` section per exercise: the problem as quoted, then the
  solution in my own words.
- Each section: the problem quoted, links to the handwritten pages, a transcription of the
  solution, review notes (mistakes made on the way), and an "In practice" line linking the result
  to ML.
- Handwritten pages are `ex-N.k-handwritten[-i].png` next to `exercises.md`, scanned small.
- The `## Status` block at the top of each `exercises.md` lists what is solved, in progress, open,
  or skipped (and why). Start there.
- A numerical exercise, or any experiment worth running, goes in a `.ipynb` next to that file.

Git commit prefix: `PRML`.
