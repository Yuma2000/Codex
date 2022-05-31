# coding: utf-8
'''
在给定的数据集划分上生成描述结果，并且计算各种评价指标
'''

from __future__ import unicode_literals
from __future__ import absolute_import
import pickle
from utils import CocoResFormat
import torch
from torch.autograd import Variable
from caption import Vocabulary
from data import get_eval_loader
from data import get_va_loader
from model import BANet
from args import vocab_pkl_path, feature_h5_path
from args import banet_pth_path, best_banet_pth_path
from args import feature_size, max_frames, max_words
from args import projected_size, hidden_size, mid_size
from args import test_range, test_prediction_txt_path, test_reference_txt_path
from args import use_cuda
from args import device
import sys
sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import DebugFunc as df

def measure(prediction_txt_path, reference):
    # 把txt格式的预测结果转换成检验程序所要求的格式
    crf = CocoResFormat()
    crf.read_file(prediction_txt_path, True)

    # df.set_trace()
    # crf.res就是格式转换之后的预测结果
    cocoRes = reference.loadRes(crf.res)
    cocoEval = COCOEvalCap(reference, cocoRes)

    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))
    return cocoEval.eval


def evaluate(vocab, banet, eval_range, prediction_txt_path, reference):
    # 载入测试数据集
    eval_loader = get_eval_loader(eval_range, feature_h5_path)
    #eval_loader = get_va_loader(eval_range, feature_h5_path)

    result = {}
    for i, (videos, video_ids) in enumerate(eval_loader):
    #for i, (videos, captions, cap_lens, video_ids) in enumerate(eval_loader):
        # 构造mini batch的Variable
        #videos = Variable(videos)
        videos = videos.float().requires_grad_(True)

        if use_cuda:
            videos = videos.to(device)

        outputs, _ = banet(videos, None)
        for (tokens, vid) in zip(outputs, video_ids):
            s = banet.decoder.decode_tokens(tokens.data)
            result[vid] = s

    prediction_txt = open(prediction_txt_path, 'w')
    for vid, s in result.items():
        prediction_txt.write('%d\t%s\n' % (vid, s))  # 注意，MSVD数据集的视频文件名从1开始

    prediction_txt.close()

    # 开始根据生成的结果计算各种指标
    metrics = measure(prediction_txt_path, reference)
    return metrics


if __name__ == '__main__':
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    # 载入预训练模型
    decoder = BANet(feature_size, projected_size, mid_size, hidden_size,
                    max_frames, max_words, vocab)
    if len(sys.argv) > 1:
        decoder.load_state_dict(torch.load(best_banet_pth_path))
    else:
        decoder.load_state_dict(torch.load(banet_pth_path))
    #decoder.cuda()
    decoder.to(device)
    decoder.eval()
    reference_json_path = '{0}.json'.format(test_reference_txt_path)
    reference = COCO(reference_json_path)
    evaluate(vocab, decoder, test_range, test_prediction_txt_path, reference)