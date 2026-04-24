cd "$(dirname "$0")/.."

# 场景是个警察办公室， 所有物体设置了碰撞属性，因此场景大
./app/python.sh gen_data.py \
--seed 0 \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/TaoBao/QAPoliceStation/OA_PoliceStation_Demo.usd \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed00_200/taobao_QAPolice_zedx01_seed00_200 \
--occupancy_resolution 0.5 \
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
--data_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed00_200/taobao_QAPolice_zedx01_seed00_200 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed00_200/taobao_QAPolice_zedx01_seed00_200/vis \
--show_num 4