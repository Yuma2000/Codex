"""
TRECVid2009 devel08 の映像に対して特徴抽出する．
ひとつの映像が対象．
2048次元になるようにConvNeXtを用いて特徴抽出
Keyフレームで画像から抽出
"""
import os, cv2, h5py, skimage
import numpy as np
import sys
import torch
from model_extract_features import ConvNextEncoder

args = sys.argv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_num = int(args[1])
key_frames_txt = "./all_frames/BG_" +str(video_num)+"_keyframes.txt"
video_name = "BG_" + str(video_num)
feature_h5_path = "./feats/tv_features.h5"
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
max_frames = 50000
feature_size = 2048

count_files = 250


# 前処理
# 呼び出し元：preprocess_frame関数
def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # シングルチャンネルのグレースケール画像を3回コピーして3チャンネルの画像にする．
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


# 前処理
# 呼び出し元：sample_frames関数
def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # ILSVRCデータセットの画像の平均（RGB形式）に基づくホワイトニング
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


# 呼び出し元：extract_features関数
def sample_frames(encoder):
    with open(key_frames_txt) as f:
        lines = f.readlines()

    for i in range(len(lines)):
        key_frame_num = int(lines[i])
        print("frame :{}".format(key_frame_num))

        image = cv2.imread('./all_frames/' + video_name + '/' + str(i) + '-' + str(key_frame_num) + '.png')
        # 前処理
        image = preprocess_frame(image)
        # 画像を2048次元の特徴量に変換
        features2048 = encoder.encode(image)  # type: torch.FloatTensor
        if i == 0:
            features_array = features2048
        else:
            features_array = torch.cat([features_array, features2048], dim=0)

    return features_array


# 特徴量をh5ファイルへ書き込む．
def extract_features(encoder):
    features_array = sample_frames(encoder)
    feats = np.zeros((count_files, feature_size), dtype="float32")
    """
    ここのpathを変えておけば現在の.h5ファイルを消さずにすむ．
    feature_h5_path = "./AllKeyVideos/"+video[:-4]+"_features.h5"
    """
    #元の特徴量を上書きしないように仮フォルダを作成．
    feature_h5_path = "./AllKeyVideos2/" +video_name+ "_features.h5"
    
    if os.path.exists(feature_h5_path):
        h5 = h5py.File(feature_h5_path, "r+")
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, "w")
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (1, count_files, feature_size),
                                          dtype="float32")
        dataset_lens = h5.create_dataset(feature_h5_lens, (1,), dtype="int")

    feats[:count_files,:] = features_array.detach().cpu().numpy()
    dataset_feats[0] = feats
    dataset_lens[0] = count_files

    h5.flush()
    h5.close()


def main():
    encoder = ConvNextEncoder()
    extract_features(encoder)
    print("--- !Extract Features Fin ---")

if __name__ == "__main__":
    main()