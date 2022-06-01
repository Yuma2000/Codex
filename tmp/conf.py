import os
import torch
from collections import namedtuple
from replayMemory import ReplayMemory

#msvd_video_root = './datasets/MSVD/youtube_videos'
#msvd_csv_path = './datasets/MSVD/MSR Video Description Corpus_refine.csv'  # 手动修改一些数据集中的错误
#msvd_video_name2id_map = './datasets/MSVD/youtube_mapping.txt'
#msvd_anno_json_path = './datasets/MSVD/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成（build_msvd_annotation.py）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = torch.float

#フレームの保存の切り替え
#save_frame = True
save_frame = False

learning_rate = 1e-3 # 1e-6
#learning_rate = 1e-6
#learning_rate = 1e-2
#learning_rate = 1e-5

#ReplayMemoryに蓄積されているものの中からBATCH_SIZE分ランダムで抽出し，Lossの計算に用いる
BATCH_SIZE = 128
#割引率みたいなもの
#GAMMA = 0.999
GAMMA = 0.999
#
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
#TARGET_UPDATE = 10
#TARGET_UPDATE = 50
#TARGET_UPDATE = 100
#TARGET_UPDATE = 500
#TARGET_UPDATE = 1
TARGET_UPDATE = 1000
#ReplayMemoryに格納するデータ量
#memory = ReplayMemory(10000)
#memory = ReplayMemory(50000)
#memory = ReplayMemory(30000)
#memory = ReplayMemory(3000)
memory = ReplayMemory(100000)


#ループ時に用いるフレーム数
max_frame = 20
#Dataloaderから取ってきたデータのバッチサイズ
batch_size = 1

#views = 3
views = 300
#views = 30

model_save = "./model_save"

model_t_path = os.path.join(model_save, 'target_DQN.pth')
model_p_path = os.path.join(model_save, 'policy_DQN.pth')

msvd_video_root = "/home/kouki/Datasets/MSVD_dataset/MSVD/youtube_videos"
msvd_csv_path = "/home/kouki/Datasets/MSVD_dataset/MSVD/MSVD_description.csv"  # 手动修改一些数据集中的错误
msvd_video_name2id_map = "/home/kouki/Datasets/MSVD_dataset/MSVD/youtube_mapping.txt"
msvd_anno_json_path = "/home/kouki/Datasets/MSVD_dataset/MSVD/annotations.json"
msvd_video_sort_lambda = lambda x: int(x[3:-4])
msvd_train_range = (0, 1200)
msvd_val_range = (1200, 1300)
msvd_test_range = (1300, 1970)



dataset = {
    'msvd': [msvd_video_root, msvd_video_sort_lambda, msvd_anno_json_path,
             msvd_train_range, msvd_val_range, msvd_test_range]
}

# 用video_root和anno_json_path这两个变量来切换所使用的数据集
# video_sort_lambda用来对视频按照名称进行排序
ds = 'msvd'
# ds = 'msr-vtt'
video_root, video_sort_lambda, anno_json_path, \
    train_range, val_range, test_range = dataset[ds]

feat_dir = 'feats'
if not os.path.exists(feat_dir):
    os.mkdir(feat_dir)

vocab_pkl_path = os.path.join(feat_dir, ds + '_vocab.pkl')
caption_pkl_path = os.path.join(feat_dir, ds + '_captions.pkl')
caption_pkl_base = os.path.join(feat_dir, ds + '_captions')
train_caption_pkl_path = caption_pkl_base + '_train.pkl'
val_caption_pkl_path = caption_pkl_base + '_val.pkl'
test_caption_pkl_path = caption_pkl_base + '_test.pkl'

feature_h5_path = os.path.join(feat_dir, ds + '_features.h5')
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
