#!/usr/bin/env bash

# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r popular.PopularUnseenRecommender \
#             -d ./data/preprocessed \
#                 --test_size 0.3 \
#                 --days 10

# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             -r popular.SegmentUnseenRecommender \
#             -fb popular.PopularRecommender \
#             --watched_pct_min 0 \
#             -d ./data/preprocessed \
#                 --test_size 0.3 \
#                 --days 10 \
#                 --segment sex age income

# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r lightfm.WeightFeaturedLightFM \
#             -fb popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --notseen_watched_upper 97 \
#                 --notseen_watched_lower 3 \
#                 --test_size 0.3 \
#                 --days 10 \
#                 --no_components 150 \
#                 --user_features_col age sex income \
#                 --item_features_col content_type release_year genres countries age_rating \
#                 --preprocess_array_split genres


# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             -r lightfm.SimpleLightFM \
#             -fb popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --test_size 0.3 \
#                 --days 10 \
#                 --watched_pct_min 0 \
#                 --no_components 50


docker run --rm -d \
    -v $(pwd)/data:/home/data/ \
    -v $(pwd)/mlruns:/home/mlruns/ \
    recsys-course \
        python -m recsys_course.train \
            --watched_pct_min 0 \
            -r hybrid.TwoStepRecommender \
            -fb popular.PopularRecommender \
            -d ./data/preprocessed \
                --models PopularRecommender \
                --models_n 100 \
                --final_moodel xgboost.XGBoostRecommender \
                --user_features_col age sex \
                --item_features_col content_type release_year \
                --category_features age sex content_type \
                --test_size 0.3 \
                --days 10