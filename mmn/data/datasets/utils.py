import os
from os.path import join, exists
import h5py
import numpy as np

import torch
import torchtext
from torch.functional import F


def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def box_iou(boxes1, boxes2):
    area1 = box_length(boxes1)
    area2 = box_length(boxes2)
    max_start = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
    min_end = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M]
    inter = (min_end - max_start).clamp(min=0)  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def box_length(boxes):
    return boxes[:, 1] - boxes[:, 0]


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()   
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores


def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d


def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            meanfeats.append(feats[s:e].mean(dim=0))
        else:
            meanfeats.append(feats[s])
    return torch.stack(meanfeats)

def maxfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    maxfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            maxfeats.append(feats[s:e].max(dim=0)[0])
        else:
            maxfeats.append(feats[s])
    return torch.stack(maxfeats)
    
def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid]['c3d_features'][:]
            else:
                feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
            vid_feats[vid] = avgfeats(feat, num_pre_clips) 
    return vid_feats

def get_vid_feat(feat_file, vid, num_pre_clips, dataset_name):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        if dataset_name == "activitynet":
            feat = f[vid]['c3d_features'][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
        elif dataset_name == "charades":
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)
        else:
            feat = f[vid][:]
            feat = F.normalize(torch.from_numpy(feat), dim=1)

    return avgfeats(feat, num_pre_clips)

def get_feat_didemo(feat_file, vid):
    assert exists(feat_file)
    with h5py.File(feat_file, 'r') as f:
        feat = f[vid][:]
    return torch.from_numpy(feat)

def get_c3d_charades(feat_file, num_pre_clips):
    assert exists(feat_file)
    feat = torch.load(feat_file)
    #feat = F.normalize(feat, dim=1)
    return maxfeats(feat, num_pre_clips)

def bert_embedding(sentence, tokenizer):
    query_token = tokenizer(sentence, return_tensors="pt", padding=True)
    word_lens = query_token['attention_mask'].sum(dim=1)
    queries = query_token['input_ids']
    return queries, word_lens


def glove_embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        vocabs.append(vocab)
    
    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)
    
    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
    return embedder(word_idxs)
