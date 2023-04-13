import torch
from pycox.models.utils import pad_col
from torch import nn
from pycox.models.loss import nll_logistic_hazard

class LossTDSurv(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def event_time_loss(self, preds, idx_durations):
        """
        Event time loss (Note: compute only for uncensored cases)
        """
        h = preds[..., 0] # batch_size, time_step

        h_l = h.gather(1, idx_durations.unsqueeze(1)).view(-1) # batch_size
        h_padded = pad_col(h, where='start') # batch_size, time_step+1
        conditional_log_haz = torch.zeros_like(h)
        for i, idx in enumerate(idx_durations):
            conditional_log_haz[i, :idx] = torch.log(1 - h_padded[i, :idx])

        L_z = -(torch.log(h_l) + conditional_log_haz.sum(1))
        return torch.mean(L_z)

    def log_event_rate(self, preds, idx_durations):
        h = preds[..., 0]  # batch_size, time_step

        h_padded = pad_col(h, where='start')  # batch_size, time_step+1
        conditional_surv_log_prob = torch.zeros_like(h)
        for i, idx in enumerate(idx_durations):
            conditional_surv_log_prob[i, :idx] = torch.log(1 - h_padded[i, :idx])

        s = conditional_surv_log_prob.sum(1)
        event_rates = torch.clamp(1-s.exp(), min=1e-8)
        log_Wt = torch.log(event_rates)
        return log_Wt

    def log_survival_rate(self, preds, idx_durations):
        h = preds[..., 0]  # batch_size, time_step

        h_padded = pad_col(h, where='start')  # batch_size, time_step+1
        conditional_surv_log_prob = torch.zeros_like(h)
        for i, idx in enumerate(idx_durations):
            conditional_surv_log_prob[i, :idx] = torch.log(1 - h_padded[i, :idx])
        log_St = conditional_surv_log_prob.sum(1)
        return log_St

    def forward(self, preds, target=None):

        idx_durations, events = target.T
        idx_durations = idx_durations.long()

        L_z = self.event_time_loss(preds[events==1.0], idx_durations[events==1.0])

        c = 1.0 - events
        log_St = self.log_survival_rate(preds, idx_durations).squeeze()
        log_Wt = self.log_event_rate(preds, idx_durations).squeeze()
        L_c = -(c * log_St + (1 - c) * log_Wt).mean(dim=0).squeeze()

        # weighted average of L_z and L_c
        loss = (self.alpha * L_z) + ((1 - self.alpha) * L_c)
        return loss
