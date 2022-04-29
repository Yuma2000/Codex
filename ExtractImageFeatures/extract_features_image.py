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
from model_extract_features import I2FEncoder
from model_extract_features import ConvNextEncoder
# import DebugFunc as df

args = sys.argv

# --- Variable declaration section --- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_root = "/home/kouki/remote-mount/tv2009/devel08/video/"
vid_num = int(args[1])
key_frame = "./all_frames/BG_" +str(vid_num)+"_keyframes.txt"

video_name = "BG_" + str(vid_num) + ".mpg"
video_path = video_root + video_name
sort_key = lambda x: int(x[3:-4])
feature_h5_path = "./feats/tv_features.h5"
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
max_frames = 50000 #250#100
feature_size = 2048

# --- Code section --- #

# extract_features関数から呼び出される．
def sample_frames(video_path, i2f, train=True):

    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print("Can not open %s." % video_path)
        pass

    frames = []
    frame_count = 0
    counter = 0

    with open(key_frame) as f:
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
            if i == int(lines[j]):
                print("frame :{}".format(lines[j]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = frame[:,:,::-1]
                frames.append(frame)
                del frame
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
                    ie_little = i2f(frame_list)

                    del frame_list

                    if counter == 0:
                        # ie_numpy = ie_little.detach().clone().cpu().numpy()
                        ie_numpy = ie_little.detach().cpu().numpy()
                    else:
                        # ie_numpy = np.concatenate([ie_numpy,ie_little.detach().clone().cpu().numpy()])
                        ie_numpy = np.concatenate([ie_numpy,ie_little.detach().cpu().numpy()])
                    del ie_little
                    counter += 1
                    frame_count += 5
    ie = torch.from_numpy(ie_numpy)
    return frame_count, ie

    """
    for i in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),1):
        if i == lines[j]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret is False:
                break
            frame = frame[:,:,::-1]
            frames.append(frame)
            j = j+1
            if len(frames) % 1 ==0:
                print("i2f {}st".format(i))
                frames = np.array(frames)
                frame_list = frames
                del frames
                frames = []
                frame_list = np.array([preprocess_frame(x) for x in frame_list])
                frame_list = frame_list.transpose((0,3,1,2))
                with torch.no_grad():
                    frame_list = torch.from_numpy(frame_list).to(device)
                torch.cuda.empty_cache()
                ie_little = i2f(frame_list)
                del frame_list

                if counter == 0:
                    # ie_numpy = ie_little.detach().clone().cpu().numpy()
                    print("A")
                    ie_numpy = ie_little.detach().cpu().numpy()
                else:
                    print("B")
                    # ie_numpy = np.concatenate([ie_numpy,ie_little.detach().clone().cpu().numpy()])
                    ie_numpy = np.concatenate([ie_numpy,ie_little.detach().cpu().numpy()])
                del ie_little
                counter += 1
            frame_count += 1
            print(frame_count)
            #ie = torch.from_numpy(ie_numpy)
        else:
            continue
        #ie = torch.from_numpy(ie_numpy)
    return frame_count, ie
    """
    """
    for i in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:,:,::-1]
        frames.append(frame)
        del frame

        if len(frames) % 5 ==0:
            print("i2f {}st".format(i))
            frames = np.array(frames)
            frame_list = frames
            del frames
            frames = []
            frame_list = np.array([preprocess_frame(x) for x in frame_list])
            frame_list = frame_list.transpose((0,3,1,2))
            with torch.no_grad():
                frame_list = torch.from_numpy(frame_list).to(device)
            torch.cuda.empty_cache()
            ie_little = i2f(frame_list)
            del frame_list

            if counter == 0:
                # ie_numpy = ie_little.detach().clone().cpu().numpy()
                ie_numpy = ie_little.detach().cpu().numpy()
            else:
                # ie_numpy = np.concatenate([ie_numpy,ie_little.detach().clone().cpu().numpy()])
                ie_numpy = np.concatenate([ie_numpy,ie_little.detach().cpu().numpy()])
            del ie_little
            counter += 1
            frame_count += 5
        #frame_count += 1
    print(frame_count)
    ie = torch.from_numpy(ie_numpy)
    return frame_count, ie
    """

# 前処理
# preprocess_frame関数から呼び出される．
def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
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
# sf関数から呼び出される．
def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # ILSVRCデータセットの画像の平均（RGB形式）に基づくホワイトニング
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


# 特徴抽出用の関数．こちらを使用している．
def extract_features(i2f):
    videos = sorted(os.listdir(video_root), key = sort_key)
    nvideos = len(videos)

    for i, video in enumerate(videos):
        if int(video[3:-4]) != vid_num:
            continue
        print("No.{} Video Name : {}".format(i,video))
        video_path = os.path.join(video_root, video)
        frame_count, ie = sample_frames(video_path, i2f, train=True)
        feats = np.zeros((frame_count, feature_size), dtype="float32")
        # feature_h5_path = "./AllKeyVideos/"+video[:-4]+"_features.h5" #ここのpathを変えておけば現在の.h5ファイルを消さずにすむ．
        feature_h5_path = "./AllKeyVideos2/" +video[:-4]+ "_features.h5" #元の特徴量を上書きしないように仮フォルダを作成．
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
    # i2f = I2FEncoder()
    i2f = ConvNextEncoder()
    # i2f.eval()  #検証用モードにする．ConvNeXtにはおそらくこの検証用モードはない．
    # i2f = torch.nn.DataParallel(i2f,device_ids=[0,1,2,3])  #GPUを複数使い，並列処理する用
    # i2f.to(device)
    # extract_feature(i2f)
    extract_features(i2f)
    print("--- !Extract Features Fin ---")

if __name__ == "__main__":
    main()




# ------------------------------------------------------------------------------- # 

def sample_frames_old(video_path, train=True):  # 現在は使われてないっぽい
    '''
    計算を減らすためにビデオフレームをサンプリングする．等間隔でmax_framesフレームを取得する．
    '''
    try:
        cap = cv2.VideoCapture(video_path)
        cap2 = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0
    num = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_count += 1
    print("# of frames = {}".format(frame_count))

    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)
    print("IDs of sampled frames: {}".format(indices))
    
    while True:
        rt, fr = cap2.read()
        if rt is False:
            break
        for i in range(len(indices)):
            if num == indices[i]:
                fr = fr[:, :, ::-1]
                frames.append(fr)
        num += 1
    frames = np.array(frames)
    frame_list = frames
    
    return frame_list, frame_count


# 特徴抽出用の関数．現在は使用していない．
def extract_feature_old(i2f):
    videos = sorted(os.listdir(video_root), key = sort_key)
    nvideos = len(videos)
    print("nvideos = {}".format(nvideos))
    ##
    if os.path.exists(feature_h5_path):
        h5 = h5py.File(feature_h5_path, "r+")
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, "w")
        """
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, max_frames, feature_size),
                                          dtype='float32')
        """
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, max_frames, feature_size),
                                          dtype="float32")
        dataset_lens = h5.create_dataset(feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print("{}: {}".format(i, video))
        video_path = os.path.join(video_root, video)
        # frame_list, frame_count = sample_frames(video_path, train=True)
        frame_count, ie = sf(video_path, i2f, train=True)
        """
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        with torch.no_grad():
            frame_list = torch.from_numpy(frame_list).to(device)
        feats = np.zeros((max_frames, feature_size), dtype='float32')
        ie = i2f(frame_list)
        """
        feats = np.zeros((frame_count, feature_size), dtype='float32')

        feats[:frame_count,:] = ie.detach().cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count
