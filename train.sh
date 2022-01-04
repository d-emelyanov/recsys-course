#!/usr/bin/env bash

# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --test_size 0.3 \
#                 --days 14 \
#                 --fb__min_watched_pct 20 \
#                 --fb__total_dur_min 7000

# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             -r popular.SegmentRecommender \
#             -fb popular.PopularRecommender \
#             --watched_pct_min 0 \
#             -d ./data/preprocessed \
#                 --test_size 0.3 \
#                 --days 10 \
#                 --segment sex age \
#                 --fb__min_watched_pct 20 \
#                 --fb__total_dur_min 2000

# docker run --rm -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r lightfm.WeightFeaturedLightFM \
#             -fb popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --notseen_watched_upper 95 \
#                 --notseen_watched_lower 5 \
#                 --test_size 0.3 \
#                 --days 10 \
#                 --no_components 150 \
#                 --user_features_col age sex \
#                 --item_features_col genres \
#                 --preprocess_array_split genres

# docker run -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
        python -m recsys_course.train \
            --watched_pct_min 10 \
            -r lightfm.WeightFeaturedLightFM \
            -fb popular.SegmentRecommender \
            -d ./data/preprocessed \
                --notseen_watched_upper 95 \
                --notseen_watched_lower 5 \
                --test_size 0.3 \
                --days 10 \
                --no_components 150 \
                --fb__min_watched_pct 10 \
                --fb__total_dur_min 2000 \
                --segment age sex


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


# docker run -d \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r hybrid.CombineRecommender \
#             -fb popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --models popular.SegmentUnseenRecommender popular.PopularUnseenRecommmender \
#                 --models_n 100 100 \
#                 --segment age sex \
#                 --test_size 0.3 \
#                 --days 10


# python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r hybrid.CombineRecommender \
#             -fb popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --models popular.SegmentUnseenRecommender popular.PopularUnseenRecommmender \
#                 --models_n 10 \
#                 --segment age sex \
#                 --test_size 0.3 \
#                 --days 10


# docker run \
#     -v $(pwd)/data:/home/data/ \
#     -v $(pwd)/mlruns:/home/mlruns/ \
#     recsys-course \
#         python -m recsys_course.train \
#             --watched_pct_min 0 \
#             -r hybrid.TwoStepRecommender \
#             -fb popular.PopularRecommender \
#             -d ./data/preprocessed \
#                 --models popular.PopularRecommender \
#                 --models_n 100 \
#                 --final_model boost.XGBoostRecommender \
#                 --user_features_col age sex \
#                 --item_features_col content_type release_year \
#                 --category_features age sex content_type \
#                 --test_size 0.3 \
#                 --days 10
