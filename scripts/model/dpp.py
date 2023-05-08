import torch
import torch.nn as nn


class DPP(nn.Module):
    def __init__(self, dpp_weight=1.0, is_seg_label_1st_sent=False):
        super().__init__()
        self.dpp_weight = dpp_weight
        self.is_seg_label_1st_sent = is_seg_label_1st_sent

    def forward(self, sents_vec, logit_pred, mask_cls, sent_sum_labels, sent_seg_labels):

        def cvt_bin_to_pos_(labels, target_val=1):
            return [[i for i, l in enumerate(label) if l == target_val] for label in labels]

        def compute_dpp_loss(L, label):
            Ly = L[label, :][:, label]
            identity = torch.eye(L.size(0), device=L.device)
            return -torch.logdet(Ly) + torch.logdet(L + identity)

        def mean_dpp_loss(batch_loss, device):
            loss_dpp = torch.tensor(batch_loss, device=device)
            # in case of nan, switch to 0
            loss_dpp[torch.isnan(loss_dpp) == 1] = 0
            return loss_dpp.mean()

        # importance
        logit_pred = logit_pred.unsqueeze(-1)  # no normalization for 1-dim
        I_mat = torch.bmm(logit_pred, logit_pred.permute(0, 2, 1))  # (B,S,S)

        # similarity
        dpp_sim = sents_vec * mask_cls[:, :, None].float()
        dpp_sim_norm = dpp_sim / torch.norm(dpp_sim, p=2, dim=-1, keepdim=True)
        S_mat = torch.bmm(dpp_sim_norm, dpp_sim_norm.permute(0, 2, 1))  # (B,S,S)
        L_mat = S_mat * I_mat

        # create L matrices that are not sigular (loss computed only from available matrices)
        L_mat = 0.5 * (L_mat + L_mat.permute(0, 2, 1))  # make symmetric matrix
        L_mat_mask = ~torch.isnan(L_mat.logdet())  # eigh is not applicable for sigular matrix
        L_mat_ = L_mat[L_mat_mask]
        e_l, v_l = torch.linalg.eigh(L_mat_)  # project L into PSD cone
        e_l[e_l < 0] = 0
        L_mat_ = torch.matmul(v_l, torch.matmul(e_l.diag_embed(), v_l.transpose(-2, -1)))

        # batch determinant computation - global level
        sum_labels = cvt_bin_to_pos_(sent_sum_labels)
        seg_labels = cvt_bin_to_pos_(sent_seg_labels)

        # DPP loss computation only for valid L matrices in a batch
        batch_loss = []
        mat_id = 0
        for i, mask in enumerate(L_mat_mask):
            if not mask:
                continue

            mat = L_mat_[mat_id]
            mat_id += 1

            sum_label = sum_labels[i]
            seg_label = seg_labels[i]
            if not self.is_seg_label_1st_sent:
                last_sent_id = next((mi for mi, mask in enumerate(mask_cls[i]) if not mask), len(mask_cls[i]))
                if seg_label and seg_label[-1] + 1 < last_sent_id:
                    seg_label.append(last_sent_id - 1)

            # get matrix with non-zero columns and rows
            L = mat[(mat != 0).any(0), :][:, (mat != 0).any(0)]

            loss = compute_dpp_loss(L, sum_label)
            batch_loss.append(loss)

        loss_dpp = mean_dpp_loss(batch_loss, L_mat.device)

        return {'loss_dpp': loss_dpp * self.dpp_weight}
