SCRIPT_DIR="/root/vepfs/isaacsim/DataGen_3dfm/scripts_zedx01_seed42_2000"

# 每项为 task_name，对应脚本为 ${SCRIPT_DIR}/${task}.sh
TASKS=(
  # Intime_Home场景
  # intime_home_000_zedx01_seed42_2000  # 已执行
  # intime_home_001_zedx01_seed42_2000
  # intime_home_002_zedx01_seed42_2000
  # intime_home_003_zedx01_seed42_2000
  # intime_home_004_zedx01_seed42_2000
  # intime_home_005_zedx01_seed42_2000
  # intime_home_006_zedx01_seed42_2000
  # intime_home_007_zedx01_seed42_2000
  # intime_home_008_zedx01_seed42_2000
  # intime_home_009_zedx01_seed42_2000
  # intime_home_010_zedx01_seed42_2000

  # Taobao场景
  # taobao_AI_vol33_scene_04_zedx01_seed42_2000  # 已执行
  # taobao_AIUE_V01_002_zedx01_seed42_2000  # 已执行
  # taobao_AIUE_V01_004_zedx01_seed42_2000  # 已执行
  # taobao_AIUE_V01_005_zedx01_seed42_2000  # 已执行
  # taobao_AIUE_V03_001_zedx01_seed42_2000  # 已执行
  # taobao_AIUE_V03_002_zedx01_seed42_2000  # 已执行
  # taobao_ModularSwimmingPool_zedx01_seed42_2000  # 已执行
  # taobao_NewScandinavian_zedx01_seed42_2000  # 已执行
)

for task in "${TASKS[@]}"; do
  ./app/python.sh submit_volcengine.py --ak "${VOLC_AK}" --sk "${VOLC_SK}" --private_image_password "${VOLC_PASSWD}" \
    --task_name "${task}" \
    --command "${SCRIPT_DIR}/${task}.sh"
done