## dataset preprocessing
python Pack_scan_files.py --scannet_data /userhome/backup_lhj/zyx/dataset/pointcloud/scannet_raw/scannet_cp_addAgg/ --data_root /userhome/backup_lhj/zyx/dataset/pointcloud/data_for_eda/scannet_others_processed/

## SR3D 
# train
sh scripts/train_sr3d.sh
sh scripts/train_sr3d_1gpu.sh

## ScanRefer 
# train
sh scripts/train_scanrefer.sh
sh scripts/train_scanrefer_single.sh
sh scripts/train_scanrefer_single_2gpu.sh
sh scripts/train_scanrefer_2gpu.sh
sh scripts/train_scanrefer_bertlite.sh
# test
sh scripts/test_scanrefer.sh
sh scripts/test_scanrefer_single.sh
sh scripts/test_scannet.sh
# measure latency
sh scripts/measure_latency_scanrefer.sh

## Nr3D 
# train
sh scripts/train_nr3d_2gpu.sh

#########
###git###
#########
## 每次git命令必须手敲，不能复制！
## 最重要的事：1.时刻清楚自己是否处于工作分支; 2.已修改代码是否提交
git branch --list                # 查看当前仓库有哪些分支，以及正处在哪个分支
git checkout RES/spformer        # 切换到工作分支
git status                       # 查看当前代码修改状态、提交状态

## 提交修改
git add . 
git commit -m "xxxxx"

## 创建工作分支。创建新分支前要确保已经提交了修改。
git checkout main                # 切换到主分支
git pull origin main             # 拉取主分支的最新更新
git branch RES/spformer          # 创建工作分支
git checkout RES/spformer        # 切换到工作分支
git branch RES/spformer/spunet-backbone # 创建工作分支

## 如果远程仓库上还没有这个工作分支（例如你刚刚在本地创建了这个分支），你可能需要先将本地分支推送到远程仓库并设置追踪关系：
git push -u origin RES/spformer
git push -u origin RES/spformer/spunet-backbone

## 日常与远程仓库保持同步
git checkout main                # 切换到主分支
git pull origin main             # 拉取主分支的最新更新

git checkout RES/spformer        # 切换到工作分支
git pull origin RES/spformer     # 拉取工作分支的最新更新

git add .
git commit -m "xxxxx"            # 合并前确保当前修改已经提交

git merge main                   # 合并主分支的更新到工作分支
git push origin RES/spformer     # 推送更新到远程工作分支
