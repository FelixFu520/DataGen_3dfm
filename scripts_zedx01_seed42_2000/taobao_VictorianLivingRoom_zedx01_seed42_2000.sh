cd "$(dirname "$0")/.."

./app/python.sh gen_data.py \
--seed 42 \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/TaoBao/VictorianLivingRoom/DemoScene.usd \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm/taobao_VictorianLivingRoom_zedx01_seed42_2000 \
--occupancy_resolution 0.1 \
--num_points 20 \
--num_paths 100 \
--max_angle_deviation 10.0 \
--erode_iterations 1 \
--wall_dilate_iterations 1 \

./app/python.sh show_data.py \
--data_dir /root/vepfs/isaacsim/workdir/3dfm/taobao_VictorianLivingRoom_zedx01_seed42_2000 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm/taobao_VictorianLivingRoom_zedx01_seed42_2000/vis \
--show_num 4

