SCRIPT_DIR="/root/vepfs/isaacsim/DataGen_3dfm/scripts/scripts_zedx01_seed00_200"

# 每项为 task_name，对应脚本为 ${SCRIPT_DIR}/${task}.sh
TASKS=(
  # Intime_Home场景
  intime_factory_000_zedx01_seed00_200
  # intime_home_000_zedx01_seed00_200
  # intime_home_001_zedx01_seed00_200
  # intime_home_002_zedx01_seed00_200
  # intime_home_003_zedx01_seed00_200
  # intime_home_004_zedx01_seed00_200
  # intime_home_005_zedx01_seed00_200
  # intime_home_006_zedx01_seed00_200
  # intime_home_007_zedx01_seed00_200
  # intime_home_008_zedx01_seed00_200
  # intime_home_009_zedx01_seed00_200
  # intime_home_010_zedx01_seed00_200

  # Taobao场景
  # taobao_AI_vol33_scene_04_zedx01_seed00_2000  # 已 executed
  # taobao_AIUE_V01_002_zedx01_seed00_2000  # 已执行
  # taobao_AIUE_V01_004_zedx01_seed00_2000  # 已 executed
  # taobao_AIUE_V01_005_zedx01_seed00_2000  # 已 executed
  # taobao_AIUE_V03_001_zedx01_seed00_2000  # 已 executed
  # taobao_AIUE_V03_002_zedx01_seed00_2000  # 已 executed
  # taobao_ModularSwimmingPool_zedx01_seed00_2000  # 已 executed
  # taobao_NewScandinavian_zedx01_seed00_2000  # 已 executed
  # taobao_NightClub_zedx01_seed00_2000  # 已 executed
  # taobao_OfficeMeetingRoom2_zedx01_seed00_2000  # 已 executed
  # taobao_Old_Laboratory2_zedx01_seed00_2000  # 已 executed
  # taobao_OutdoorFurniture_zedx01_seed00_2000 # 已 executed
  # taobao_ParkingGarage_zedx01_seed00_2000 # 已 executed
  # taobao_PostSovietFlat2_zedx01_seed00_2000 # 已 executed
  # taobao_PostSovietKitchen_zedx01_seed00_2000 # 已 executed
  # taobao_QAModularParking_zedx01_seed42_2000 # 已执行
  # taobao_QAOffice_zedx01_seed00_2000 # 已 executed
  # taobao_QAPolice_zedx01_seed00_2000  # 已 executed
  # taobao_ResearchCenter_zedx01_seed00_2000 # 已 executed
  # taobao_RetroOffice_zedx01_seed00_2000 # 已 executed
  # taobao_SchoolGym_zedx01_seed00_2000 # 已 executed
  # taobao_ShootingRange_zedx01_seed00_2000 # 已 executed
  # taobao_ShowRoom_zedx01_seed00_2000 # 已 executed
  # taobao_StylizedRoom_zedx01_seed00_2000 # 已 executed
  # taobao_UtopianCity_zedx01_seed00_2000 # 已 executed
  # taobao_VictorianLivingRoom_zedx01_seed00_2000 # 已 executed

  # TaoBao02场景
)

# 切换到脚本目录
cd "$(dirname "$0")/.."

for task in "${TASKS[@]}"; do
  python submit_volcengine.py --ak "${VOLC_AK}" --sk "${VOLC_SK}" --private_image_password "${VOLC_PASSWD}" \
    --task_name "${task}" \
    --command "${SCRIPT_DIR}/${task}.sh"
done