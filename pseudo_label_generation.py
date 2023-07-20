import torch
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
import os
import  json
import argparse
import nltk

DATA_PATH={
    'charades': 'data/charades/train.json',
    'activitynet': 'data/activitynet/train.json'
}

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def nms(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed], scores[~suppressed]


def nms2(moments, scores, sims, thresh, topk=5):
    reversed_idx = torch.arange(0, scores.size(0)).long()
    final_scores = scores * sims
    final_scores, ranks = final_scores.sort(descending=True)
    # ranks = np.random.permutation(scores.shape[0])
    moments = moments[ranks]
    scores = scores[ranks]
    sims = sims[ranks]
    reversed_idx = reversed_idx[ranks]
    # return moments, scores, sims, reversed_idx
    suppressed = ranks.zero_().long()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i] >= topk:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] += 1
    suppressed = suppressed >= topk
    return moments[~suppressed], scores[~suppressed], sims[~suppressed], reversed_idx[~suppressed]


def enumerate_with_sliding_window(scores, captions, stride, nms_topk, nms_thresh, device='cpu'):
    flattened_captions = [s for c in captions for s in c]
    flattened_proposals = []
    flattened_scores = []
    flattened_similarity = []
    max_stride = scores.size(-1)
    stride = min(stride, max_stride)
    for kernel_size in range(stride, max_stride+1, stride):
        res = F.conv1d(scores.view(-1, 1, scores.size(-1)), torch.ones((1, 1, kernel_size)).to(device) / kernel_size).view(scores.size(0), scores.size(1), -1)
        res = res - 1.0*(scores.sum(dim=-1, keepdim=True) - res * kernel_size) / (scores.size(-1) - kernel_size)
        res = res.view(-1, res.size(-1))
        proposals = torch.arange(0, res.size(-1))
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1)
        for idx in range(len(flattened_captions)):
            mask = res[idx] > 0
            if idx >= len(flattened_proposals):
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(res[idx][mask])
                flattened_similarity.append(scores.flatten(0, 1)[idx].max())
            else:
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], res[idx][mask]], dim=0)

    filtered_captions = []
    filtered_proposals = []
    filtered_scores = []
    filtered_similarity = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            nms_proposals, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], nms_thresh)
            
            for k in range(min(nms_topk, nms_scores.size(0))):
                filtered_captions.append(flattened_captions[idx])
                filtered_similarity.append(flattened_similarity[idx])
                filtered_proposals.append(nms_proposals[k])
                filtered_scores.append(nms_scores[k])

    return filtered_captions, torch.stack(filtered_proposals), torch.stack(filtered_scores), torch.stack(filtered_similarity)


def enumerate_with_events(scores, captions, device='cpu'):
    with open('data/activitynet/events.pkl', 'rb') as f:
        events = pickle.load(f)

    scores = scores.flatten(0, 1)
    flattened_captions = [s for c in captions for s in c]
    flattened_scores = []
    flattened_sims = []
    max_stride = scores.size(-1)

    for event in events[vid]:
        s, e = event[0] / 128, (event[1] + 1) / 128
        mask = torch.arange(max_stride, device=device) / max_stride
        mask = (mask >= s) & (mask < e)
        mask = mask.to(float)
        if (1 - mask).sum() == 0:
            res = torch.zeros((scores.size(0),), device=device)
        else:
            res = (scores * mask).sum(dim=-1) / mask.sum() - (scores * (1 - mask)).sum(dim=-1) / (1 - mask).sum()
        flattened_scores.append(res)
        flattened_sims.append(torch.ones_like((scores).max(dim=-1)[0]))
    
    flattened_scores = torch.stack(flattened_scores, dim=-1)
    flattened_sims = torch.stack(flattened_sims, dim=-1)
    max_idx = flattened_scores.argmax(dim=-1)

    filtered_captions = []
    filtered_proposals = []
    filtered_scores = []
    filtered_sims = []
    for idx in range(len(flattened_captions)):
        filtered_captions.append(flattened_captions[idx])
        filtered_proposals.append(events[vid][int(max_idx[idx])])
        filtered_scores.append(flattened_scores[idx, max_idx[idx]])
        filtered_sims.append(flattened_sims[idx, max_idx[idx]])

    return filtered_captions, torch.tensor(filtered_proposals), torch.tensor(filtered_scores), torch.tensor(filtered_sims)


def generate_proposal(vid, stride, nms_thresh=0.3, nms_topk=1, device='cpu', args=None):
    caption_path = os.path.join(args.caption_path, vid+'.pkl')
    caption_feature_path = os.path.join(args.caption_feat_path, vid+'.npy')
    video_feature_path = os.path.join(args.video_feat_path, vid+'.npy')

    try:
        with open(caption_path, 'rb') as f:
            captions = pickle.load(f)
        with open(caption_feature_path, 'rb') as f:
            caption_features = np.load(f)
        with open(video_feature_path, 'rb') as f:
            video_features = np.load(f)
    except:
        return [], [], [], []
    
    verb_filt = torch.zeros((len(captions), len(captions[0])))
    for i in range(len(captions)):
        for j in range(len(captions[i])):
            flag = False
            for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(captions[i][j])):
                if 'VB' in tag:
                    if word not in ['is', 'am', 'are', 'was', 'were', 'being', 'been', 'to be', 'be']:
                        flag = True
                        break
            if flag:
                verb_filt[i][j] = 1
    
    v1 = F.normalize(torch.tensor(caption_features).view(-1, caption_features.shape[-1]), dim=-1).to(device)
    v2 = F.normalize(torch.tensor(video_features), dim=-1).to(device)
    scores = (v1 @ v2.T).reshape(caption_features.shape[0], caption_features.shape[1], -1)
    scores = scores / 2 + 0.5
    scores = scores * verb_filt.unsqueeze(-1)

    if args.dataset == 'charades':
        return enumerate_with_sliding_window(scores, captions, stride, nms_topk, nms_thresh, device)
    elif args.dataset == 'activitynet':
        return enumerate_with_events(scores, captions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='charades', type=str, choices=['charades', 'activitynet'])
    parser.add_argument('--video_feat_path', required=True, type=str)
    parser.add_argument('--caption_feat_path', required=True, type=str)
    parser.add_argument('--caption_path', required=True, type=str)
    parser.add_argument('--num_stnc', default=10, type=int)
    parser.add_argument('--stride', default=20, type=int)
    parser.add_argument('--prop_topk', default=3, type=int)
    parser.add_argument('--stnc_th', default=0.7, type=float)
    parser.add_argument('--stnc_topk', default=3, type=int)
    args = parser.parse_args()

    with open(DATA_PATH[args.dataset]) as f:
        data = json.load(f)

    new_data = []
    topk = args.num_stnc
    for vid in tqdm(data.keys()):
        captions, proposals, scores, sims = generate_proposal(vid, stride=args.stride, nms_topk=args.prop_topk, device='cpu', args=args)
        if (len(captions) == 0):
            continue
        nms_proposals, nms_scores, nms_sims, reversed_idx = nms2(proposals, scores, sims, thresh=args.stnc_th, topk=args.stnc_topk)
        for idx in range(min(topk, nms_proposals.size(0))):
            new_data.append([vid, data[vid]['duration'], (nms_proposals[idx] * data[vid]['duration']).tolist(), captions[reversed_idx[idx]]])

    if args.dataset == 'charades':
        with open('EMB/data/dataset/charades/charades_sta_train_pseudo.txt', 'w') as f:
            for vid, duration, (s, t), query in new_data:
                print('%s %.2f %.2f##%s'%(vid, s, t, query.strip()), file=f)
    elif args.dataset == 'activitynet':
        tmp = {}
        for vid, duration, (s, t), query in new_data:
            if vid not in tmp:
                tmp[vid] = {'duration': duration, 'timestamps': [], 'sentences': []} 
            tmp[vid]['timestamps'].append([s / 127, t / 127])
            tmp[vid]['sentences'].append(query)
        with open('EMB/data/dataset/activitynet/train_pseudo.json', 'w') as f:
            json.dump(tmp, f)
