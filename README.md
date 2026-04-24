# DataGen_3dfm
为3dfm项目生成仿真数据

## 安装isaacsim
### 安装isaacsim和对应资产
安装目录建议安装在$HOME目录, isaacsim和资产可以选择一个版本安装，不必全部安装

官方推荐安装到$HOME/isaacsim文件夹下，但是我这个文件已经创建了，我就安装到$HOME/isaac_sim中了

```
mkdir -p $HOME/isaac_sim
cd $HOME/isaac_sim


# 下载软件和资产
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-1-4.5.0.zip
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-2-4.5.0.zip
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-3-4.5.0.zip
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.001
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.002
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.1.0.zip.003


# 合并资产
cat isaac-sim-assets-1-4.5.0.zip  isaac-sim-assets-2-4.5.0.zip isaac-sim-assets-3-4.5.0.zip > isaac-sim-assets-4.5.0.zip
cat isaac-sim-assets-complete-5.1.0.zip.001 isaac-sim-assets-complete-5.1.0.zip.002 isaac-sim-assets-complete-5.1.0.zip.003 > isaac-sim-assets-5.1.0.zip


# 解压
unzip -d 4.5/ isaac-sim-standalone-4.5.0-linux-x86_64.zip
unzip -d 5.1/ isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip -d 4.5_asset isaac-sim-assets-4.5.0.zip
unzip -d 5.1_asset isaac-sim-assets-5.1.0.zip


# git管理, 有时候会误修改isaacsim软件中代码，所以用git管理下，如果有变化，可以及时发现
cd $HOME/isaac_sim/4.5/
git init
git add .
git commit -m "init"

cd $HOME/isaac_sim/4.5_asset/
git init
git add .
git commit -m "init"

cd $HOME/isaac_sim/5.1/
git init
git add .
git commit -m "init"

cd $HOME/isaac_sim/5.1_asset/
git init
git add .
git commit -m "init"

```
### 配置isaacsim默认资产
isaacsim资产可以从网络和本地加载，上一步已经下载了资产，现在配置下，让isaacsim可以识别到
#### 5.1 配置
参考：https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html

编辑apps/isaacsim.exp.base.kit文件，根据自身情况修改

```
[settings]
persistent.isaac.asset_root.default = "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1"

exts."isaacsim.gui.content_browser".folders = [
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Robots",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/People",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/IsaacLab",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Props",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Environments",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Materials",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Samples",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Sensors",
]

exts."isaacsim.asset.browser".folders = [
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Robots",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/People",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/IsaacLab",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Props",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Environments",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Materials",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Samples",
    "/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1/Isaac/Sensors",
]

```
## 安装本项目
```
cd $HOME
git clone git@github.com:FelixFu520/DataGen_3dfm.git
cd $HOME/DataGen_3dfm

# 链接isaacsim, 我这里链接的是isaacsim5.1，可以根据需要选择不同的版本
ln -s $HOME/isaac_sim/5.1 app   # 链接isaacsim5.1
ln -s $HOME/isaac_sim/5.1_asset asset_internal  # 链接isaacsim5.1资产
ln -s $HOME/isaac_sim/asset_extern asset_extern # 链接生成的场景资产

# 安装一些库
./app/python.sh -m pip install loguru cupy-cuda11x plyfile
```
## 生成&可视化数据
```
# 启动Isaacsim
./app/isaac-sim.sh \
--/persistent/isaac/asset_root/default=/home/fufa/isaac_sim/5.1_asset/Assets/Isaac/5.1

# 生成数据
./app/python.sh gen_data.py \
--scene_usd_url /home/fufa/d-isaacsim/asset_extern/interior_template_20251211/interior_template.usdc \
--camera_usd_url /home/fufa/projects2026/DataGen_3dfm/assets/zedx01.usd \
--output_dir /home/fufa/projects2026/DataGen_3dfm/workdir/3dfm/home000_zedx01 \
--num_points 4 \
--num_paths 1 

# 可视化数据
./app/python.sh show_data.py \
--data_dir /home/fufa/projects2026/DataGen_3dfm/workdir/3dfm/home000_zedx01 \
--save_dir /home/fufa/projects2026/DataGen_3dfm/workdir/3dfm/home000_zedx01/vis
```
## 相机说明
- `assets/ZED_X.usdc`: isaacsim官方的相机
- `assets/zedx01.usd`: 在`assets/ZED_X.usdc`修改，让左右相机向分别向中心旋转8度
- `assets/zedx02.usd`: 在`assets/ZED_X.usdc`修改，让左右相机向分别向非中心旋转8度, 同时修改分辨率
- `assets/zedx03.usd`: 在`assets/ZED_X.usdc`修改，扩展成4个相机
## 火山上跑
### 封装镜像
```
docker pull nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# 在docker的CUDA官方镜像中，这些内容要重新安装
apt-get update && apt-get install -y libgl1 libxt6 libxml2 libxrandr2 libxinerama1 libxcursor1 libxi6 libvulkan1 zenity  libglu1-mesa
```
### 提交任务
```
bash submit.sh
```