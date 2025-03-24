from commons import *     
import dtw_kernel_v4 as dtw_kernel
import dtw_kernel_mm_v4 as dtw_kernel_mm
import dtw_kernel_dba
import dtw_kernel_mm_dba

import dtw_kernel_fp_v4 as dtw_kernel_fp

import dtw_kernel_mm_fp_v4 as dtw_kernel_mm_fp

def _call_dtw_runner(k_feats, k_feat_weights, multi_s_feats, runner, use_fixed_path, feat_batch_size=128):
    if not use_fixed_path:
        args = [k_feats.contiguous(), k_feat_weights.contiguous(), [s.contiguous() for s in multi_s_feats]]
    else:
        args = [k_feats[0].contiguous(), k_feat_weights[0].contiguous(), k_feats[1].contiguous(), k_feat_weights[1].contiguous(), [s.contiguous() for s in multi_s_feats]]

    args.append([sample.shape[0] for sample in multi_s_feats])
    args.extend(args[0].shape)
    args.append(len(multi_s_feats))
    k_feat_slope, k_feat_intercept = \
    torch.zeros(args[-1], args[-4], args[-3], args[-2], device=args[0].device), \
    torch.zeros(args[-1], args[-4], args[-3], args[-2], device=args[0].device)
    runner(*args, k_feat_slope, k_feat_intercept, feat_batch_size)
    return k_feat_slope, k_feat_intercept

def call_dtw_kernel(k_feats, k_feat_weights, multi_s_feats, use_mm, use_fixed_path, feat_batch_size=128):
    return _call_dtw_runner(k_feats, k_feat_weights, multi_s_feats, (dtw_kernel_mm.run if not use_fixed_path else dtw_kernel_mm_fp.run) if use_mm else (dtw_kernel.run if not use_fixed_path else dtw_kernel_fp.run), use_fixed_path, feat_batch_size)

def call_dtw_kernel_dba(k_feats, k_feat_weights, multi_s_feats, use_mm, feat_batch_size=128):
    return _call_dtw_runner(k_feats, k_feat_weights, multi_s_feats, dtw_kernel_mm_dba.run if use_mm else dtw_kernel_dba.run, False, feat_batch_size)

