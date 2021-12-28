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

python -m recsys_course.test \
            --watched_pct_min 0 \
            -r lightfm.WeightFeaturedLightFM \
            -fb popular.PopularRecommender \
            -t ./data/raw/sample_submission.csv \
            -s ./data/submission.csv \
            -d ./data/preprocessed \
                --notseen_watched_upper 95 \
                --notseen_watched_lower 5 \
                --days 10 \
                --no_components 150 \
                --user_features_col age sex income \
                --item_features_col content_type release_year genres countries age_rating \
                --preprocess_array_split genres