cd "$(dirname "$0")/.."

./app/python.sh gen_data.py \
--seed 43 \
--scene_usd_url /root/vepfs/isaacsim/DataGen_3dfm/asset_extern/Intime_Home/home_000/interior_template.usdc \
--camera_usd_url /root/vepfs/isaacsim/DataGen_3dfm/assets/zedx01.usd \
--output_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed43_2000/home000_zedx01_seed43_2000 \
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
--data_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed43_2000/home000_zedx01_seed43_2000 \
--save_dir /root/vepfs/isaacsim/workdir/3dfm_zedx01_seed43_2000/home000_zedx01_seed43_2000/vis \
--show_num 4