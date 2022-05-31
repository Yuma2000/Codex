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
import DebugFunc as df

# --- Variable declaration section --- #
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpus = (0,1,2,3)
device = torch.device(f"cuda:{min(gpus)}"if len(gpus)>0 else "cpu")
# device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
video_root = "/home/kouki/remote-mount/tv2009/devel08/video"
sort_key = lambda x: int(x[3:-4])
feature_h5_path = "./feats/tv_features.h5"
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
max_frames = 50000 #250#100
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

def sf(video_path, i2f, train=True):

    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print("Can not open %s." % video_path)
        pass

    frames = []
    frame_count = 0
    counter = 0

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
        # feats = np.zeros((max_frames, feature_size), dtype='float32')
        feats = np.zeros((frame_count, feature_size), dtype='float32')

        feats[:frame_count,:] = ie.detach().cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count

def extract_feature_v2(i2f):
    videos = sorted(os.listdir(video_root), key = sort_key)
    nvideos = len(videos)

    for i, video in enumerate(videos):
        print("No.{} Video Name : {}".format(i,video))
        video_path = os.path.join(video_root, video)
        frame_count, ie = sf(video_path, i2f, train=True)
        feats = np.zeros((frame_count, feature_size), dtype="float32")
        print("Crear exfeats!")
        feature_h5_path = "./feats/"+video[:-4]+"_features.h5"
        if os.path.exists(feature_h5_path):
            h5 = h5py.File(feature_h5_path, "r+")
            dataset_feats = h5[feature_h5_feats]
            dataset_lens = h5[feature_h5_lens]
        else:
            h5 = h5py.File(feature_h5_path, "w")
            dataset_feats = h5.create_dataset(feature_h5_feats,
                                             (1, frame_count, feature_size),
                                             # (nvideos, frame_count, feature_size),
                                             dtype="float32")
            dataset_lens = h5.create_dataset(feature_h5_lens, (1,), dtype="int")# (nvideos,), dtype="int")

        feats[:frame_count,:] = ie.detach().cpu().numpy()
        # dataset_feats[i] = feats
        # dataset_lens[i] = frame_count
        dataset_feats[0] = feats
        dataset_lens[0] = frame_count

        h5.flush()
        h5.close()
        #del daraset_feats
        #del dataset_lens
        #del feats
        #del frame_count


def main():
    i2f = I2FEncoder()
    i2f.eval()
    i2f = torch.nn.DataParallel(i2f,device_ids=[0,1,2,3])
    i2f.to(device)
    # extract_feature(i2f)
    extract_feature_v2(i2f)
    print("--- Extract Features Fin ---")

if __name__ == "__main__":
    main()
