# CLAUDE.md

Instructions for agents working in this repo.

## What this repo is

This repo holds my machine-learning course and book work: lecture material, labs, homeworks,
exams, and my own solutions. Each course or book gets its own folder. Existing folders use
`course-<name>-<institution/lecturer>` (e.g. `course-ml-voron`) or `book-<title>`
(e.g. `book-An-Introduction-to-Statistical-Learning`). Inside a folder, subfolders use whatever
names the source itself uses: `lectures/`, `labs/`, `homeworks/`, `chNN_<topic>/`, and so on.

Solutions go in the same subfolder as the material they solve. The active book course,
`book-pattern-recognition-and-machine-learning-bishop/`, uses one `exercises.md` per chapter
(see its `README.md`).

Commit messages start with the subproject prefix listed in `README.md` (for example `PRML: ...`).

## Don't invent progress that isn't verified

The older folders (`statistical-ml`, `Neural-networks`, `nlp`, `course-ml-voron`,
`course-ml2-tartu-swenlaur`, the ISL book) were collected over time. Nobody has checked their
completion state file by file. Don't assume something is finished just because the file exists.

## Agent memory does not live here

This repo holds course material and my own solutions, nothing else. Agent memory lives in the
KnowledgeNexus workspace, in `memory/ml-tutor.md`: the study plan, session log, next actions,
and the tutor's teaching rules. This repo is not a submodule there; KnowledgeNexus refers to it
by name.
