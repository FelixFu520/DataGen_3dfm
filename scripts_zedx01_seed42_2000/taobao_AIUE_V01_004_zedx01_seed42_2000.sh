cd "$(dirname "$0")/.."

# 场景是两层别墅，因此场景大
./app/python.sh gen_data.py \
--seed 42 \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/TaoBao/AIUE_V01_004/AIUE_V01_004.usd \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_AIUE_V01_004_zedx01_seed42_2000 \
--occupancy_resolution 0.25 \
--num_points 100 \
--num_paths 20 \
--max_angle_deviation 10.0 \
--erode_iterations 2 \
--obstacle_dilate_iterations 2 \
--obstacle_envelope_iterations 2 \
--step_size_xy 0.3 \
--step_size_z 0.1 \
--max_dz_per_step 0.1

./app/python.sh show_data.py \
--data_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_AIUE_V01_004_zedx01_seed42_2000 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_AIUE_V01_004_zedx01_seed42_2000/vis \
--show_num 4