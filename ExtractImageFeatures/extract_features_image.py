"""
TRECVid2009 devel08 の映像に対して特徴抽出する．
ひとつの映像が対象．
2048次元になるようにResNet50を用いて特徴抽出
Keyフレームで画像から抽出
"""
import os, cv2, h5py, skimage
import numpy as np
import sys
import torch
from model_extract_features import ResNet50Encoder
from model_extract_features import ConvNextEncoder

args = sys.argv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# video_root = "/home/kouki/remote-mount/tv2009/devel08/video/"  ###
video_num = int(args[1])
key_frames_txt = "./all_frames/BG_" +str(video_num)+"_keyframes.txt"

# video_name = "BG_" + str(video_num) + ".mpg"  ###
video_name_new = "BG_" + str(video_num)
# video_path = video_root + video_name  #どこにあるどのビデオかを特定する．  ###
sort_key = lambda x: int(x[3:-4])
feature_h5_path = "./feats/tv_features.h5"
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
max_frames = 50000 #250#100
feature_size = 2048


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
def sample_frames(video_path, encoder, train=True):

    """
    ビデオは使わないのでこの例外処理はコメントアウトしておく．
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print("Can not open %s." % video_path)
        pass
    """

    frames = []
    frame_count = 0
    counter = 0

    with open(key_frames_txt) as f:
        lines = f.readlines()
    for i in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),1):
        #ret, frame = cap.read()
        #if ret is False:
        #    break
        for j in range(len(lines)):
            #cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            #ret, frame = cap.read()
            #if ret is False:
            #    break
            #print(frame)
            #print(len(lines))
            if i == int(lines[j]):  #キーフレームの時
                print("frame :{}".format(lines[j]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  #i番目のフレームからスタート
                ret, frame = cap.read()

                #ビデオの最後までいった時
                if ret is False:
                    break
                
                frame = frame[:,:,::-1]
                frames.append(frame)
                del frame

                #フレームが5つ溜まったら
                if len(frames) % 5 ==0:
                    frames = np.array(frames)
                    frame_list = frames
                    del frames
                    frames = []
                    frame_list = np.array([preprocess_frame(x) for x in frame_list])
                    frame_list = frame_list.transpose((0,3,1,2))

                    with torch.no_grad():
                        frame_list = torch.from_numpy(frame_list).to(device)
                    torch.cuda.empty_cache()
                    ie_little = encoder(frame_list)

                    del frame_list

                    if counter == 0:
                        ie_numpy = ie_little.detach().cpu().numpy()
                    else:
                        ie_numpy = np.concatenate([ie_numpy,ie_little.detach().cpu().numpy()])
                    del ie_little
                    counter += 1
                    frame_count += 5
    ie = torch.from_numpy(ie_numpy)
    return frame_count, ie


# 特徴量をh5ファイルへ書き込む．
def extract_features(encoder):
    videos = sorted(os.listdir(video_root), key = sort_key)
    nvideos = len(videos)  #動画の本数

    # 全てのビデオでforを回す．
    for i, video in enumerate(videos):
        # 使うビデオでなければ飛ばす．
        if int(video[3:-4]) != vid_num:
            continue
        print("No.{} Video Name : {}".format(i,video))
        # video_path = os.path.join(video_root, video)  ###
        frame_count, ie = sample_frames(video_path, encoder, train=True)
        feats = np.zeros((frame_count, feature_size), dtype="float32")
        """
        ここのpathを変えておけば現在の.h5ファイルを消さずにすむ．
        feature_h5_path = "./AllKeyVideos/"+video[:-4]+"_features.h5" 
        """
        #元の特徴量を上書きしないように仮フォルダを作成．
        feature_h5_path = "./AllKeyVideos2/" +video[:-4]+ "_features.h5"
        if os.path.exists(feature_h5_path):
            h5 = h5py.File(feature_h5_path, "r+")
            dataset_feats = h5[feature_h5_feats]
            dataset_lens = h5[feature_h5_lens]
        else:
            h5 = h5py.File(feature_h5_path, "w")
            dataset_feats = h5.create_dataset(feature_h5_feats,
                                             (1, frame_count, feature_size),
                                             dtype="float32")
            dataset_lens = h5.create_dataset(feature_h5_lens, (1,), dtype="int")

        feats[:frame_count,:] = ie.detach().cpu().numpy()
        dataset_feats[0] = feats
        dataset_lens[0] = frame_count

        h5.flush()
        h5.close()


def main():
    # encoder = ResNet50Encoder()
    encoder = ConvNextEncoder()
    # encoder.eval()  #検証用モードにする．ConvNeXtにはおそらくこの検証用モードはない．
    # GPUを複数使い，並列処理する．
    # encoder = torch.nn.DataParallel(encoder,device_ids=[0,1,2,3])  
    # encoder.to(device)
    extract_features(encoder)
    print("--- Extract Features Fin ---")

if __name__ == "__main__":
    main()