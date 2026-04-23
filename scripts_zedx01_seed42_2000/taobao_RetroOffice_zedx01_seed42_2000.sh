cd "$(dirname "$0")/.."

# 场景是个小的办公室，有挡板， 所有物体设置了碰撞属性，因此场景大
./app/python.sh gen_data.py \
--seed 42 \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/TaoBao/RetroOffice/Demonstation.usd \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_RetroOffice_zedx01_seed42_2000 \
--occupancy_resolution 0.1 \
--num_points 100 \
--num_paths 20 \
--max_angle_deviation 10.0 \
--erode_iterations 2 \
--obstacle_dilate_iterations 2 \
--obstacle_envelope_iterations 0 \
--step_size_xy 0.3 \
--step_size_z 0.1 \
--max_dz_per_step 0.1

./app/python.sh show_data.py \
--data_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_RetroOffice_zedx01_seed42_2000 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed42_2000/taobao_RetroOffice_zedx01_seed42_2000/vis \
--show_num 4