```
python -m recsys_course.train --data data/preprocessed/interactions.csv data/preprocessed/users.csv data/preprocessed/items.csv -r popular.simple.PopularRecommender --params -k 10 -days 30
```