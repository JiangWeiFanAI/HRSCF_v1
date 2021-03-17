import torch
import torch.nn as nn
import torch.nn.functional as func
class crps_loss(nn.Module):
    def __init__(self):
        super(crps_loss_1, self).__init__()

    
    
    def forward(self,observations,forecasts):

        forecasts=forecasts.squeeze().permute(1,2,0) 
        observations=observations.squeeze()
#         forecasts=forecasts.sort(dim=-1)[0]
        if observations.ndim == forecasts.ndim - 1:
            # sum over the last axis
            assert observations.shape == forecasts.shape[:-1]

        #     observations = observations[..., np.newaxis]
            observations=observations.unsqueeze(-1)

        #     score = np.nanmean(abs(forecasts - observations), -1)
            score=torch.mean(torch.abs(forecasts - observations),dim=-1)
            # insert new axes along last and second to last forecast dimensions so

            # forecasts_diff expands with the array broadcasting
        #     forecasts_diff = (np.expand_dims(forecasts, -1) -
        #                       np.expand_dims(forecasts, -2))
            forecasts_diff=(forecasts.unsqueeze(-1) -
                            forecasts.unsqueeze(-2))


        #     score += -0.5 * np.nanmean(abs(forecasts_diff),
        #                                    axis=(-2, -1))
            score += -0.5 * torch.mean(torch.abs(forecasts_diff),
                                           dim=(-2, -1))
            return torch.mean(score)
        elif observations.ndim == forecasts.ndim:
            # there is no 'realization' axis to sum over (this is a deterministic
            # forecast)
            return torch.abs(observations - forecasts)