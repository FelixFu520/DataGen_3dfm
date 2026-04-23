cd "$(dirname "$0")/.."

# 场景是个海边场景， 但是只把房子设置了碰撞属性，因此场景大
./app/python.sh gen_data.py \
--seed 42 \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/TaoBao/AIUE_V01_002/AIUE_V01_002.usd \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_AIUE_V01_002_zedx01_seed42_2000 \
--occupancy_resolution 0.1 \
--num_points 100 \
--num_paths 20 \
--max_angle_deviation 10.0 \
--erode_iterations 2 \
--wall_dilate_iterations 2 \
--step_size_xy 0.3 \
--step_size_z 0.1 \
--max_dz_per_step 0.1

./app/python.sh show_data.py \
--data_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_AIUE_V01_002_zedx01_seed42_2000 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_AIUE_V01_002_zedx01_seed42_2000/vis \
--show_num 4