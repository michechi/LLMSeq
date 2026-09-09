> Historical notes retained for context. For the current TMLR work, start with [the repository guide](../README.md) and [TMLR status](tmlr_status.md). Commands and relative paths below refer to the old layout and are not the current protocol.

# General Scope
Thin is my git-hub repository for the paper I recently submitted to ICML26 and where I've got some reviews to it (you can find them in @Rebuttals_ICML26-2.pdf). You can find the paper in @/paper/paper.tex. 

# Models-Data-files
## Data
I know this git repository is Huge, so feel free to ask me whenever you want info-questions-clarifications. We are working toghether here. Now, for the data you can find all here: @/data/tested/ and tricky is the csv 9. 

## Models
We are testing different models here belonging to different type of familes. I particular we have:
* BERT
* RoBERTa
* LSTM
* LLama-1B/8B
* Qwen3-4B-think
* Qwen2.5-14B

## Scripts
We are working on a HPC cluster so we run with slurm jobs. You can find them in @/scripts/slurm/ 
* for the LLMs we are using @LLM_fraction_experiments.sh which uses the LLM_fraction_experiment.py file (note: for now we want to use always the 100% of the data).
* for the DL baselines we are using @DL_baselines.slurm (which uses the @DL_TR_baselines_experiment.py). Note: in these files we have a lot of different models, but we want to use LSTM, RNN-Transformer and Transformer
* k-gram baseline: you can find the code in @/simulation/stat_test.py

# Today's goal:
Day 1 should be entirely about one thing: quantifying how much of Tricky (/data/simulation/tested/*_9.csv) is solvable from order-invariant content alone, how much from short local order, and how much only from richer sequence modeling. That is the cleanest direct response to rUwY’s main objection, and it fits my current revision, which already narrowed the theory and softened the parity claims. Reviewer rUwY explicitly asked for a more quantitative gap between “semantic/content-like” and “sequential” understanding, while my current draft still mainly contrasts k-grams against full models without a true bag-of-events control.

## By the end of the day I want three concrete artifacts:
* A signal decomposition table for Tricky deterministic and Tricky random, with parity as a sanity contrast.
* A matched-histogram counterfactual table showing whether models separate positive from negative sequences when the full 26-dimensional count vector is identical.
* One reviewer-facing paragraph that says, numerically, how much signal is bag/content, how much is local order, and how much your models recover.