# Sentence level quality estimation

This repository is for the [Sentence-level QE task 2020 Codalab competition](https://competitions.codalab.org/competitions/22831).
To recreate our results one can run the EnglishToChinese notebook.
A baseline notebook has been provided as well.

## Task description

The shared task on Quality Estimation aims to examine automatic methods for estimating the quality of machine translation output at run-time, without relying on reference translations.This variant looks at sentence-level prediction, where participating systems are required to score each translated sentence according to direct assessments (DA) on their quality. The DA score is a number in 0-100 which has been given by humans, where 0 is the lowest possible quality and 100 is a perfect translation. For the task, we collected three annotations per instance and use the average of a z-standardised (per annotator) version of the raw score

## Scores

Our best model is an RNN using GRU cells (saved as modelGRU) which scored the following on the held out dataset:
- Pearson: 0.3001
- MAE: 0.7024
- RMSE: 0.8887
