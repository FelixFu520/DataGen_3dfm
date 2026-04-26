SCENE_USD_URL=/root/vepfs/isaacsim/DataGen_3dfm/asset_extern/TaoBao02/AIUE_V02_001/AIUE_V02_001_2.usd
CAMERA_USD_URL=/root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd
OUTPUT_DIR=/root/vepfs/isaacsim/workdir/3dfm_zedx01_seed02_100/taobao02_AIUE_V02_001_zedx01_seed02_100
OCCUPANCY_RESOLUTION=0.25
NUM_POINTS=10
NUM_PATHS=10
MAX_ANGLE_DEVIATION=10.0
ERODE_ITERATIONS=1
OBSTACLE_DILATE_ITERATIONS=1
OBSTACLE_ENVELOPE_ITERATIONS=10
STEP_SIZE_XY=0.3
STEP_SIZE_Z=0.1
MAX_DZ_PER_STEP=0.1

# 切换到脚本目录
cd "$(dirname "$0")/../.."

# 创建软链接，让isaacsim可以识别到资产
ln -s /root/vepfs/isaacsim/5.1_asset /root/5.1_asset

# 生成数据
./app/python.sh gen_data.py \
--seed 1 \
--scene_usd_url $SCENE_USD_URL \
--camera_usd_url $CAMERA_USD_URL \
--output_dir $OUTPUT_DIR \
--occupancy_resolution $OCCUPANCY_RESOLUTION \
--num_points $NUM_POINTS \
--num_paths $NUM_PATHS \
--max_angle_deviation $MAX_ANGLE_DEVIATION \
--erode_iterations $ERODE_ITERATIONS \
--obstacle_dilate_iterations $OBSTACLE_DILATE_ITERATIONS \
--obstacle_envelope_iterations $OBSTACLE_ENVELOPE_ITERATIONS \
--step_size_xy $STEP_SIZE_XY \
--step_size_z $STEP_SIZE_Z \
--max_dz_per_step $MAX_DZ_PER_STEP

# 可视化数据
./app/python.sh show_data.py \
--data_dir $OUTPUT_DIR \
--save_dir $OUTPUT_DIR/vis \
--show_num 4