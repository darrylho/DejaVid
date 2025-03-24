from commons import *

from collections import Counter
import dba


class CUDADTWKernelWorkerDist:
    @staticmethod
    def worker_fn(gpu_id, use_mm, use_fixed_path, feat_batch_size, job_queue, result_queue):
        from utils import call_dtw_kernel
        torch.cuda.set_device(gpu_id)
        while True:
            args = job_queue.get()
            # if args is None:  # Sentinel to shut down the worker
            #     break
            ret = call_dtw_kernel(*args, use_mm, use_fixed_path, feat_batch_size=feat_batch_size)
            del args
            torch.cuda.empty_cache()
            result_queue.put(ret)
    
    def __init__(self, use_mm, device_id, feat_batch_size = 128, use_fixed_path = False):
        # self.num_workers = num_workers if num_workers is not None else torch.cuda.device_count()
        # import torch.multiprocessing as mp
        from torch.multiprocessing import Queue 
        self.job_queue, self.result_queue = Queue(), Queue() 
        self.process = torch.multiprocessing.Process(target=DTWKernelWorkerDist.worker_fn, args=(device_id, use_mm, use_fixed_path, feat_batch_size, self.job_queue, self.result_queue))
        self.process.start()
    
    def run(self, k_feats, k_feat_weights, multi_s_feats):
        self.job_queue.put((k_feats, k_feat_weights, multi_s_feats))
        return self.result_queue.get()

    def __del__(self):
        self.process.terminate()
        self.process.join()

class NaiveDTWKernelWorkerDist:
    def __init__(self, use_mm, device_id, feat_batch_size = 128, use_fixed_path = False):
        assert use_fixed_path, "naive dtw kernel worker only supports fixed path"
        assert not use_mm, "naive dtw kernel worker only supports non-mm"
        self.feat_batch_size = feat_batch_size
    
    def run(self, _k_feats, _k_feat_weights, multi_s_feats):
        assert len(_k_feats) == len(_k_feat_weights) == 2
        fixed_k_feats, k_feats = _k_feats
        fixed_feat_weights, k_feat_weights = _k_feat_weights

        with torch.no_grad():
            multi_calced_dist_offset = []
            for s_feats in multi_s_feats:
                calced_dist = torch.zeros(fixed_k_feats.size(0), fixed_k_feats.size(1), s_feats.size(0), device=fixed_k_feats.device)
                for fl in range(0, s_feats.size(-1), self.feat_batch_size):
                    fr = min(fl + self.feat_batch_size, s_feats.size(-1))
                    lhs = fixed_feat_weights[:,:,fl:fr].unsqueeze(2).expand(-1, -1, s_feats.size(0), -1) * \
                            torch.abs(
                                fixed_k_feats[:,:,fl:fr].unsqueeze(2).expand(-1, -1, s_feats.size(0), -1) - 
                                s_feats[:, fl:fr].unsqueeze(0).unsqueeze(0).expand(fixed_k_feats.size(0), fixed_k_feats.size(1), -1, -1))
                    calced_dist += lhs.sum(-1)
                multi_calced_dist_offset.append(calced_dist)
        
            k_feat_slope = torch.zeros(len(multi_s_feats), k_feats.size(0), k_feats.size(1), k_feats.size(2), device=k_feats.device)
            k_feat_intercept = torch.zeros(len(multi_s_feats), k_feats.size(0), k_feats.size(1), k_feats.size(2), device=k_feats.device)
            # mean_series.shape = num_kernels, num_timesteps, num_feats_per_timestep

            def attribute_slope_intercept(samp_i, kid, kt, st):
                s_val_vec, k_val_vec = multi_s_feats[samp_i][st], k_feats[kid, kt]
                mult_vec = torch.where(s_val_vec < k_val_vec, -1.0, 1.0)
                k_feat_slope[samp_i, kid, kt] -= mult_vec
                k_feat_intercept[samp_i, kid, kt] += mult_vec * s_val_vec


            for samp_i in range(len(multi_s_feats)):
                s_feats = multi_s_feats[samp_i]
                _calced_dist = multi_calced_dist_offset[samp_i]
                for kid in range(k_feats.size(0)):
                    calced_dist_kid = _calced_dist[kid]
                    cost_mat = np.zeros((k_feats.size(1), s_feats.size(0)))
                    cost_mat[0, 0] = calced_dist_kid[0, 0]
                    for i in range(1, k_feats.size(1)):
                        cost_mat[i, 0] = cost_mat[i-1, 0] + calced_dist_kid[i, 0]
                    for i in range(1, s_feats.size(0)):
                        cost_mat[0, i] = cost_mat[0, i-1] + calced_dist_kid[0, i]

                    for kt in range(1, k_feats.size(1)):
                        for st in range(1, s_feats.size(0)):
                            cost_mat[kt, st] = calced_dist_kid[kt, st] + min(
                                cost_mat[kt-1, st],
                                cost_mat[kt, st-1],
                                # cost_mat[kt-1, st-1] if kt > 0 and st > 0 else float('inf')
                            )

                    kt, st = k_feats.size(1)-1, s_feats.size(0)-1
                    while kt > 0 or st > 0:
                        attribute_slope_intercept(samp_i, kid, kt, st)
                        if kt > 0 and st > 0:
                            if cost_mat[kt-1, st] < cost_mat[kt, st-1]:
                                kt -= 1
                            else:
                                st -= 1
                        elif kt > 0:
                            kt -= 1
                        else:
                            st -= 1

                    assert kt == 0 and st == 0
                    attribute_slope_intercept(samp_i, kid, kt, st)
            
            return k_feat_slope, k_feat_intercept

DTWKernelWorkerDist = CUDADTWKernelWorkerDist
# DTWKernelWorkerDist = NaiveDTWKernelWorkerDist


class DTWModelDist(nn.Module):
    def __init__(self, mean_series, use_mm, device_id, feat_batch_size = 128, weights_path = None, use_fixed_path = False, w_one_init = False):
        super().__init__()
        self.mean_series = nn.Parameter(mean_series.to(f'cuda:{device_id}'))
        
        # mean_series.shape = num_kernels, num_timesteps, num_feats_per_timestep
        if not w_one_init:
            self.feat_log_weights = nn.Parameter(torch.randn_like(mean_series, device=f'cuda:{device_id}') if weights_path == None else torch.load(weights_path, map_location=f'cuda:{device_id}'))
        else: # exp(0)=1
            self.feat_log_weights = nn.Parameter(torch.zeros_like(mean_series, device=f'cuda:{device_id}') if weights_path == None else torch.load(weights_path, map_location=f'cuda:{device_id}'))

        self.worker = DTWKernelWorkerDist(use_mm, device_id, feat_batch_size=feat_batch_size, use_fixed_path=use_fixed_path)

        self.use_fixed_path = use_fixed_path

        if use_fixed_path:
            self.register_buffer('fixed_feat_weights', self.feat_log_weights.exp().detach().clone().contiguous())
            self.register_buffer('fixed_k_feats', self.mean_series.detach().clone().contiguous())

    def fixed_reset(self):
        self.fixed_feat_weights *= 0
        self.fixed_feat_weights += self.feat_log_weights.exp().detach()
        self.fixed_k_feats *= 0
        self.fixed_k_feats += self.mean_series.detach()

    def forward(self, multi_s_feats):
        k_feats, k_feat_weights = self.mean_series, self.feat_log_weights.exp()
        if len(multi_s_feats) == 0:
            return torch.zeros(0, len(k_feats), device=k_feats.device)
        k_feat_slope, k_feat_intercept = \
            self.worker.run(
                k_feats.detach() if not self.use_fixed_path else [self.fixed_k_feats, k_feats.detach()],
                k_feat_weights.detach().clone() if not self.use_fixed_path else [self.fixed_feat_weights, k_feat_weights.detach().clone()],
                multi_s_feats)
        
        feat_dist_multi = k_feat_weights * (k_feats * k_feat_slope + k_feat_intercept)
        # feat_dist_multi.shape = num_samples x num_kernels x num_timesteps x num_feats_per_timestep

        feat_dist = torch.sum(feat_dist_multi, dim=(2,3))
        # feat_dist.shape = num_samples x num_kernels

        logits = -feat_dist
        # logits.shape = num_samples x num_kernels

        return logits 



