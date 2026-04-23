SCRIPT_DIR="/root/vepfs/isaacsim/DataGen_3dfm/scripts"

# 每项为 task_name，对应脚本为 ${SCRIPT_DIR}/${task}.sh
TASKS=(
  taobao_AIUE_V03_001_zedx01_seed42_2000        # AIUE_V03_001, 场景较大, 可以多跑些数据
  taobao_AIUE_V03_002_zedx01_seed42_2000        # AIUE_V03_002, 场景较大, 可以多跑些数据
  taobao_AI_vol33_scene_04_zedx01_seed42_2000   # AI_vol33_scene_04, 场景较大, 可以多跑些数据
  taobao_ModularSwimmingPool_zedx01_seed42_2000 # ModularSwimmingPool, 场景适中
  taobao_NewScandinavian_zedx01_seed42_2000     # NewScandinavian, 场景较小
  taobao_NightClub_zedx01_seed42_2000           # NightClub, 场景较大
  taobao_OfficeMeetingRoom2_zedx01_seed42_2000  # OfficeMeetingRoom2, 场景小
  taobao_Old_Laboratory2_zedx01_seed42_2000     # Old_Laboratory2, 场景小
  taobao_OutdoorFurniture_zedx01_seed42_2000    # OutdoorFurniture, 场景中
  taobao_ParkingGarage_zedx01_seed42_2000       # ParkingGarage, 场景大
  taobao_PostSovietFlat2_zedx01_seed42_2000     # PostSovietFlat2, 场景小
  taobao_PostSovietKitchen_zedx01_seed42_2000   # PostSovietKitchen, 场景小
  taobao_QAModularParking_zedx01_seed42_2000    # QAModularParking, 场景大
  taobao_QAOffice_zedx01_seed42_2000            # QAOffice, 场景中
  taobao_QAPolice_zedx01_seed42_2000            # QAPolice, 场景大
)

for task in "${TASKS[@]}"; do
  ./app/python.sh submit_volcengine.py --ak "${VOLC_AK}" --sk "${VOLC_SK}" --private_image_password "${VOLC_PASSWD}" \
    --task_name "${task}" \
    --command "${SCRIPT_DIR}/${task}.sh"
done
