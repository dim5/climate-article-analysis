# Analyzer

This folder contains Python scripts used for cleaning, processing, and analyzing the collected articles.

To run the scripts, you need to install the dependencies from `environment.yml` and activate the environment:

```bash
conda env create -f environment.yml -n analyzer
conda activate analyzer
```

See the [conda docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for further details.

## Structure

The scripts are in the `src` folder, while their in/outputs need to be put in the `data` folder.
Unfortunately, due to possible legal reasons, I cannot share the collected articles.

The `src/pre-process` contains scripts related to pre-processing, `src/process` contains the named entity extraction, clustering, and graph creator files, and `src/analysis` a script that can be a starting point for exploring the results.

## Named entity merge rules

I also extracted some named entity merge rules (`res\entity_merge_list_labeled.txt` & `res\entity_merge_list_unlabeled.txt`).
They follow the following format: `target | [source list]` with one target entity per line.

The two rule files are complementary.

## Notes

The scripts are the results of exploratory data analysis.
