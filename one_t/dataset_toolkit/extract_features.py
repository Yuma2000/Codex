"""
TRECVid2009 devel08 の映像に対して特徴抽出する
ひとつの映像が対象である
2048次元になるようにResNet50を用いて特徴抽出
5フレームごとに抽出される
"""
# --- Import section --- #
import os, cv2, h5py, skimage
import numpy as np
import torch
from model_extract_features import I2FEncoder

# --- Variable declaration section --- #
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
video_root = "/home/kouki/remote-mount/tv2009/devel08/video"
sort_key = lambda x: int(x[3:-4])
feature_h5_path = "./feats/tv_features.h5"
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
max_frames = 100
feature_size = 2048

# --- Code section --- #
def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
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
    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)
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

def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
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

def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # 根据在ILSVRC数据集上的图像的均值（RGB格式）进行白化
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def extract_feature(i2f):
    videos = sorted(os.listdir(video_root), key = sort_key)
    nvideos = len(videos)
    if os.path.exists(feature_h5_path):
        h5 = h5py.File(feature_h5_path, "r+")
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, "w")
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, max_frames, feature_size),
                                          dtype='float32')
        dataset_lens = h5.create_dataset(feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print("{}: {}".format(i, video))
        video_path = os.path.join(video_root, video)
        #
        frame_list, frame_count = sample_frames(video_path, train=True)
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        with torch.no_grad():
            frame_list = torch.from_numpy(frame_list).to(device)
        feats = np.zeros((max_frames, feature_size), dtype='float32')
        ie = i2f(frame_list)
        feats[:frame_count,:] = ie.detach().cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count
        # if i == 0:
        #     break

def main():
    i2f = I2FEncoder()
    i2f.eval()
    #i2f.to(device)
    i2f = torch.nn.DataParallel(i2f)
    i2f.to(device)
    extract_feature(i2f)
    print("--- Extract Features Fin ---")

if __name__ == "__main__":
    main()
