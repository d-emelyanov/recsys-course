```
python -m recsys_course.train -r popular.simple.PopularRecommender -d data/preprocessed/ -n 10 -o --optuna_trials 1 --days__type int --days__low 2 --days__high 7
```


```
python -m recsys_course.test -r popular.simple.PopularRecommender -d data/preprocessed/ --test_file data/raw/sample_submission.csv --submission data/submission.csv -n 10 --days 10
```