#!/usr/bin/env bash

# python -m recsys_course.test \
#     -r popular.SegmentRecommender \
#     -fb popular.PopularRecommender \
#     -d ./data/preprocessed \
#     -t ./data/raw/sample_submission.csv \
#     -s ./data/submission.csv \
#         --days 10 \
#         --watched_pct_min 0 \
#         --segment age sex


# python -m recsys_course.test \
#     -t ./data/raw/sample_submission.csv \
#     -s ./data/submission.csv \
#     -r popular.SegmentRecommender \
#     -fb popular.PopularRecommender \
#     --watched_pct_min 5 \
#     -d ./data/preprocessed \
#         --days 10 \
#         --segment sex age \
#         --fb__min_watched_pct 20 \
#         --fb__total_dur_min 2000

python -m recsys_course.test \
            --watched_pct_min 0 \
            -r hybrid.TwoStageRecommender \
            -fb popular.PopularRecommender \
            -t ./data/raw/sample_submission.csv \
            -s ./data/submission.csv \
            -d ./data/preprocessed \
                --days 10 \
                --models popular.SegmentRecommender \
                --models_n 100 \
                --models_w 1 \
                --final_model_sample 0.3 \
                --final_model boost.CatboostRecommender \
                --features score_0 \
                --category_features age sex release_year_cat content_type \
                --text_features countries \
                --segment age sex \
                --fb__min_watched_pct 10 \
                --fb__total_dur_min 2000