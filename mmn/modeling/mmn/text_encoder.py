import torch
from torch import nn
from transformers import DistilBertModel


class DistilBert(nn.Module):
    def __init__(self, joint_space_size, dataset):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc_out1 = nn.Linear(768, joint_space_size)
        self.fc_out2 = nn.Linear(768, joint_space_size)
        self.dataset = dataset
        self.layernorm = nn.LayerNorm(768)
        self.aggregation = "avg"  # cls, avg

    def forward(self, queries, wordlens):
        '''
        Average pooling over bert outputs among words to be sentence feature
        :param queries:
        :param wordlens:
        :param vid_avg_feat: B x C
        :return: list of [num_sent, C], len=Batch_size
        '''
        sent_feat = []
        sent_feat_iou = []
        for query, word_len in zip(queries, wordlens):  # each sample (several sentences) in a batch (of videos)
            N, word_length = query.size(0), query.size(1)
            attn_mask = torch.zeros(N, word_length, device=query.device)
            for i in range(N):
                attn_mask[i, :word_len[i]] = 1  # including [CLS] (first token) and [SEP] (last token)
            bert_encoding = self.bert(query, attention_mask=attn_mask)[0]  # [N, max_word_length, C]  .permute(2, 0, 1)
            if self.aggregation == "cls":
                query = bert_encoding[:, 0, :]  # [N, C], use [CLS] (first token) as the whole sentence feature
                query = self.layernorm(query)
                out_iou = self.fc_out1(query)
                out = self.fc_out2(query)
            elif self.aggregation == "avg":
                avg_mask = torch.zeros(N, word_length, device=query.device)
                for i in range(N):
                    avg_mask[i, :word_len[i]] = 1       # including [CLS] (first token) and [SEP] (last token)
                avg_mask = avg_mask / (word_len.unsqueeze(-1))
                bert_encoding = bert_encoding.permute(2, 0, 1) * avg_mask  # use avg_pool as the whole sentence feature
                query = bert_encoding.sum(-1).t()  # [N, C]
                query = self.layernorm(query)
                out_iou = self.fc_out1(query)
                out = self.fc_out2(query)
            else:
                raise NotImplementedError
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou


def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.MMN.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME
    return DistilBert(joint_space_size, dataset_name)
