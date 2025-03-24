from commons import *
from utils import call_dtw_kernel, call_dtw_kernel_dba

def resample_time_series(time_series_list, target_length):
    resampled_list = []
    for tensor in time_series_list:
        # Interpolate
        resampled_tensor = F.interpolate(
            tensor.unsqueeze(0).transpose(1, 2),
            size=target_length,
            mode='linear',
            align_corners=True
        ).transpose(1, 2).squeeze(0)
        resampled_list.append(resampled_tensor.clone())
    return resampled_list

def get_samples_per_class(dataloader_train, sample_size_per_class, target_length):
    samples_per_class, population_per_class = dict(), dict()

    for padded_batch, batch_lens, labels in tqdm(dataloader_train):
        for i in range(padded_batch.size(0)):
            label = labels[i].item()
            if label not in samples_per_class:
                samples_per_class[label] = []
                population_per_class[label] = 0
            samples_per_class[label].append(padded_batch[i, :batch_lens[i], :].detach().clone())
            population_per_class[label] += 1
            if len(samples_per_class[label]) == sample_size_per_class + 1: # Reservoir Sampling
                j = np.random.randint(0, population_per_class[label])
                if j < sample_size_per_class:
                    samples_per_class[label][j] = samples_per_class[label][-1]
                samples_per_class[label].pop(-1)
    for label in samples_per_class.keys():
        samples_per_class[label] = resample_time_series(samples_per_class[label], target_length)
        torch.cuda.empty_cache()
    return samples_per_class

def get_approx_medoid_per_class(samples_per_class, use_mm, feat_batch_size=128):
    # Approximate Medoid Calculation
    medoid_per_class = dict()
    for label in tqdm(samples_per_class.keys()):
        samples = [t.clone() for t in samples_per_class[label]]
        distance_mat = torch.zeros(len(samples), len(samples))
        for i in range(len(samples)): # can div by 2
            kern = samples[i].unsqueeze(0).clone()
            k_feat_slope, k_feat_intercept = \
                call_dtw_kernel(kern,
                                torch.ones_like(kern),
                                samples,
                                use_mm,
                                use_fixed_path=False,
                                feat_batch_size=feat_batch_size)
            distance_mat[i] = (kern * k_feat_slope + k_feat_intercept).sum(dim=(1,2,3))
        medoid_idx = distance_mat.sum(dim=0).argmin().item()
        medoid_per_class[label] = samples[medoid_idx]
    return medoid_per_class

def DBA_update(center, samples, use_mm, feat_batch_size=128):
    # DBA Algorithm
    kern = center.unsqueeze(0).clone()

    k_feat_count, k_feat_sum = \
    call_dtw_kernel_dba(kern,
                    torch.ones_like(kern),
                    samples,
                    use_mm,
                    feat_batch_size=feat_batch_size)
    # k_feat_count, k_feat_sum shape = num_samples x num_kernels x num_timesteps x num_feats_per_timestep

    return k_feat_sum.sum(dim=(0,1)) / k_feat_count.sum(dim=(0,1)) # shape = num_timesteps x num_feats_per_timestep
    
def calculate_mean_series(dataloader_train, use_mm, sample_size_per_class=50, max_iter=100, feat_batch_size=128, target_length=64):
    samples_per_class = get_samples_per_class(dataloader_train, sample_size_per_class, target_length)
    
    medoid_per_class = get_approx_medoid_per_class(samples_per_class, use_mm, feat_batch_size=feat_batch_size)

    for label in tqdm(samples_per_class.keys()):
        samples = [t.clone() for t in samples_per_class[label]]
        medoid = medoid_per_class[label]
        for it in range(max_iter):
            medoid = DBA_update(medoid, samples, use_mm, feat_batch_size=feat_batch_size)
        medoid_per_class[label] = medoid
    
    
    return torch.stack([medoid_per_class[label] for label in sorted(medoid_per_class.keys())], dim=0).clone()
    # num_kernels, num_timesteps, num_feats_per_timestep
    


