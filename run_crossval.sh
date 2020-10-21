#!/bin/bash
# Compute metrics for all test files with different gamma and h_res
python spatun_crossval.py --config kyle_ransalu/1_toy1_vel --mode tq
python spatun_crossval.py --config kyle_ransalu/2_carla1_vel --mode tq
python spatun_crossval.py --config kyle_ransalu/3_astyx1_vel --mode tq
python spatun_crossval.py --config kyle_ransalu/4_nuscenes1_vel --mode tq
python spatun_crossval.py --config kyle_ransalu/5_airsim1_pos --mode tq
python spatun_crossval.py --config kyle_ransalu/5_airsim1_vel --mode tq
python spatun_crossval.py --config kyle_ransalu/6_jfk_partial1_pos --mode tq
python spatun_crossval.py --config kyle_ransalu/6_jfk_partial1_vel --mode tq

