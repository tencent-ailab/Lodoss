import torch.nn as nn
import numpy as np
import evaluate

from .pl_model import Summarizer
from ..utils import print_stats_core, print_rank_0


class LongDocSummarizer(Summarizer):
    def __init__(self, args, model, tokenizer):
        super(LongDocSummarizer, self).__init__(args, model)
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.rouge_eval = evaluate.load(f'{self.args.cache_eval_dir}/rouge') # , use_stemmer=True)

    def forward(self, input_ids, cls_ids, mask_cls, sent_sum_labels, sent_seg_labels):
        return self.model(input_ids, cls_ids, mask_cls, sent_sum_labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        loss = outputs['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True,
                 logger=True, batch_size=self.args.batch_size)
        return loss
    
    def get_prediction_ids(self, outputs, mask_cls, sent_sum_labels):
        def cvt_bin_to_pos_(labels, target_val=1):
            return [[i for i, l in enumerate(label) if l == target_val] for label in labels]

        if self.args.oracle_inf:
            # summary generation based on oracle labels
            oracle_labels = (sent_sum_labels * mask_cls.float()).cpu().data.numpy()
            oracle_id_batch = cvt_bin_to_pos_(oracle_labels)
            preds = oracle_id_batch
        else:
            # summary generation based on summarization prediction scores
            sigmoid = nn.Sigmoid()
            logit_sum = sigmoid(outputs['logit_sum'])
            logit_sum = logit_sum + mask_cls.float()
            logit_sum = logit_sum.cpu().data.numpy()
            preds = np.argsort(-logit_sum, 1)
        return preds
    
    def get_prediction_text(self, preds_sum_ids, input_text, ref_summary):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i: i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        preds, refs = [], []
        for pi, (pids, in_text, summary) in enumerate(zip(preds_sum_ids, input_text, ref_summary)):
            _pred = []
            if not in_text:
                continue
            
            sent_limit = len(summary) if self.args.num_sent_inf == -1 else self.arg.num_sent_inf
            
            _pred_pid = []
            for pid in pids:
                if pid >= len(in_text):
                    continue
                cand_sent = in_text[pid].strip()
                if self.args.block_trigram:
                    if not _block_tri(cand_sent, _pred):
                        _pred.append(cand_sent)
                        _pred_pid.append(pid)
                else:
                    _pred.append(cand_sent)
                    _pred_pid.append(pid)

                if len(_pred) == sent_limit:
                    break

            if self.args.sort_sum_pred:
                sorted_preds = sorted([(sent, idx) for sent, idx in zip(_pred, _pred_pid)], key=lambda x: x[1])
                _pred = [sent for sent, idx in sorted_preds]

            refs.append(summary)
            preds.append(_pred)
        return preds, refs

    def common_step(self, batch):
        outputs = self.forward(*batch[:-3])
        loss = outputs['loss']

        input_ids, cls_ids, mask_cls, sent_sum_labels, input_text, summary, index = batch
        preds_sum_ids = self.get_prediction_ids(outputs, mask_cls, sent_sum_labels)
        preds, refs = self.get_prediction_text(preds_sum_ids, input_text, summary)
        return loss, preds, refs

    def common_epoch(self, outputs, split='val'):
        total_loss, total_norm = 0, 0
        preds_sent_cnt, preds_word_cnt = [], []
        for loss, predictions, references in outputs:
            total_loss += loss
            total_norm += 1
            n_sent, n_word = self.compute_metrics(predictions, references)
            preds_sent_cnt += n_sent
            preds_word_cnt += n_word
        avg_loss = total_loss / total_norm
        log = dict()
        log[f"{split}_loss"] = avg_loss

        n_sent = print_stats_core(preds_sent_cnt, 'preds_sent_cnt')
        for k, v in n_sent.items():
            log[f'{split}_sent_' + k] = v
        n_word = print_stats_core(preds_word_cnt, 'preds_word_cnt')
        for k, v in n_word.items():
            log[f'{split}_word_' + k] = v

        rouge_scores = self.rouge_eval.compute(use_stemmer=True)
        for k, v in rouge_scores.items():
            log[k] = v
        log['rouge_avg'] = (rouge_scores['rouge1'] + rouge_scores['rouge2'] + rouge_scores['rougeLsum']) / 3
        self.log_dict(log)
        print_rank_0(log)
        return {'log': log}

    def compute_metrics(self, prediction, reference):
        preds_sent_cnt = [len(sents) for sents in prediction]
        preds_word_cnt = [len(' '.join(sents).split()) for sents in prediction]
        decoded_preds = ["\n".join(pred) for pred in prediction]
        decoded_labels = ["\n".join(ref) for ref in reference]

        self.rouge_eval.add_batch(predictions=decoded_preds, references=decoded_labels)
        return preds_sent_cnt, preds_word_cnt

    def validation_step(self, batch):
        return self.common_step(batch)

    def validation_epoch_end(self, outputs):
        return self.common_epoch(outputs, 'val')
    
    def test_step(self, batch):
        return self.common_step(batch)

    def test_epoch_end(self, outputs):
        return self.common_epoch(outputs, 'test')
