cd "$(dirname "$0")/.."

./app/python.sh gen_data.py \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/Scene-Home-Issac/home_001/home_001.usdc \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm/home001_zedx01 \
--occupancy_resolution 0.1 \
--num_points 20 \
--num_paths 100 \
--max_angle_deviation 10.0 \
--erode_iterations 2 \
--wall_dilate_iterations 2 \

./app/python.sh show_data.py \
--data_dir /root/vepfs/isaacsim/workdir/3dfm/home001_zedx01 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm/home001_zedx01/vis \
--show_num 4

