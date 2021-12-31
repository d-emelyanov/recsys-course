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
            -r lightfm.WeightFeaturedLightFM \
            -fb popular.PopularRecommender \
            -t ./data/raw/sample_submission.csv \
            -s ./data/submission.csv \
            -d ./data/preprocessed \
                --notseen_watched_upper 95 \
                --notseen_watched_lower 5 \
                --days 10 \
                --no_components 150 \
                --fb__min_watched_pct 10 \
                --fb__total_dur_min 2000 \