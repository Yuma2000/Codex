# --- Import section --- #
import torch
import torch.utils.data as data
import h5py

# --- Variable declaration section --- #
feature_h5_path = "../dataset_toolkit/feats/tv_features.h5"
feature_h5_feats = 'feats'
# --- Code section --- #
class VideoDataset(data.Dataset):
    
    def __init__(self, eval_range, feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5_file = h5py.File(feature_h5, 'r')
        self.video_feats = h5_file[feature_h5_feats]

    def __getitem__(self, index):
        video_id = self.eval_list[index]
        # print("video_id={}".format(video_id))
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, video_id

    def __len__(self):
        return len(self.eval_list)

#shuffle=True
def get_eval_loader(cap_pkl, feature_h5, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
    vd = VideoDataset(cap_pkl, feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader