
#include <vector>
#include <utility>
#include <numeric>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <ctime>
#include <fstream>
#include <type_traits>
#include <utility> 


#include <torch/extension.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using namespace std;
namespace py = pybind11;


inline __device__ __host__ int64_t calcIndex(const int32_t& size1, const int& index1) {
    return (int64_t)index1;
}
inline __device__ __host__ int64_t calcIndex(const int32_t& size1, const int& index1, const int32_t& size2, const int& index2) {
    return index1*(int64_t)size2+index2;
}
inline __device__ __host__ int64_t calcIndex(const int32_t& size1, const int& index1, const int32_t& size2, const int& index2, const int32_t& size3, const int& index3) {
    return (index1*(int64_t)size2+index2)*size3+index3;
}
inline __device__ __host__ int64_t calcIndex(const int32_t& size1, const int& index1, const int32_t& size2, const int& index2, const int32_t& size3, const int& index3, const int32_t& size4, const int& index4) {
    return ((index1*(int64_t)size2+index2)*size3+index3)*size4+index4;
}
inline __device__ __host__ int64_t calcIndex(const int32_t& size1, const int& index1, const int32_t& size2, const int& index2, const int32_t& size3, const int& index3, const int32_t& size4, const int& index4, const int32_t& size5, const int& index5) {
    return (((index1*(int64_t)size2+index2)*size3+index3)*size4+index4)*size5+index5;
}

#define CALC_WEIGHT_DIST(w, k_val, s_val) (w*fabs((k_val)-(s_val)))

#ifndef FOR_DBA
    #define ATTRIBUTE_SLOPE_INTERCEPT(k_val, s_val, blob_offset, fi) do{\
        int mult = (s_val)<(k_val)?-1:1;\
        atomicAdd(&k_feat_slope_blob[(blob_offset)+(fi)], (-mult));\
        atomicAdd(&k_feat_intercept_blob[(blob_offset)+(fi)], (mult)*(s_val));\
    }while(0)
#else 
    #define ATTRIBUTE_SLOPE_INTERCEPT(k_val, s_val, blob_offset, fi) do{\
        atomicAdd(&k_feat_slope_blob[(blob_offset)+(fi)], 1);\
        atomicAdd(&k_feat_intercept_blob[(blob_offset)+(fi)], (s_val));\
    }while(0)
#endif 

inline __device__ void set_intm_ptrs(
    float* k_feat_weights_ptr, 
    float* k_added_feats_ptr, float* s_added_feats_ptr, 
    int32_t num_kernels, 
    int32_t num_k_timesteps, 
    int32_t num_added_feats,
    int32_t num_samples,    
    int kid, 
    int32_t num_s_timesteps,
    float* &k_feat_weights_intm_ptr, 
    float* &k_added_feats_intm_ptr, float* &s_added_feats_intm_ptr
){
    k_feat_weights_intm_ptr = &k_feat_weights_ptr[calcIndex(num_kernels, kid, num_k_timesteps, 0, num_added_feats/*+1*/, 0)];
    k_added_feats_intm_ptr = &k_added_feats_ptr[calcIndex(num_kernels, kid, num_k_timesteps, 0, num_added_feats, 0)];
    s_added_feats_intm_ptr = &s_added_feats_ptr[calcIndex(num_s_timesteps, 0, num_added_feats, 0)];
}

inline __device__ float get_dist_intm(
    const float* __restrict__ k_feat_weights_intm_ptr, 
    const float* __restrict__ k_added_feats_intm_ptr, const float* __restrict__ s_added_feats_intm_ptr, 
    int32_t num_kernels, 
    int32_t num_k_timesteps, 
    int32_t num_added_feats,
    int32_t num_samples,    
    int kt, int st,
    int32_t num_s_timesteps
) {
    float dist = 0.0;        
    for(int fi = 0; fi < num_added_feats; fi++) {
        dist += CALC_WEIGHT_DIST(
            k_feat_weights_intm_ptr[calcIndex(num_k_timesteps, kt, num_added_feats/*+1*/, fi)],
            k_added_feats_intm_ptr[calcIndex(num_k_timesteps, kt, num_added_feats, fi)],
            s_added_feats_intm_ptr[calcIndex(num_s_timesteps, st, num_added_feats, fi)]);
    }
    return dist;
}


inline __device__ void attribute_var_intm(
    float* k_added_feats_intm_ptr, float* s_added_feats_intm_ptr, 
    int32_t num_kernels, 
    int32_t num_k_timesteps, 
    int32_t num_added_feats,
    int32_t num_samples,    
    int samp_i, int kid, int kt, int st,
    float* k_feat_slope_blob, float* k_feat_intercept_blob,
    int32_t num_s_timesteps,
    int64_t blob_offset
) {
    for(int fi = 0; fi < num_added_feats; fi++) {
        ATTRIBUTE_SLOPE_INTERCEPT(
            k_added_feats_intm_ptr[calcIndex(num_k_timesteps, kt, num_added_feats, fi)], 
            s_added_feats_intm_ptr[calcIndex(num_s_timesteps, st, num_added_feats, fi)], 
            blob_offset, fi);
    }
}


template<int THR_CNT, int batch_size>
__global__ void per_samp_i_kid_kt(
    float* k_feat_weights_ptr, 
    float* k_added_feats_ptr, float** cuda_multi_s_added_feats_ptr, 
    float** cuda_multi_calced_dist_ptr,
    float* k_feat_slope_blob,
    float* k_feat_intercept_blob,
    int32_t num_kernels, 
    int32_t num_k_timesteps, 
    int32_t num_added_feats,
    int32_t num_samples,
    int64_t *cuda_multi_num_s_timesteps, 
    float* multi_tmp3,
    int32_t max_num_s_timesteps,
    int32_t samp_i_offset
) {
    int32_t samp_i = (blockIdx.x * blockDim.x + threadIdx.x) + samp_i_offset;
    int32_t kid = blockIdx.z * blockDim.z + threadIdx.z;
    
    float *s_added_feats_ptr = cuda_multi_s_added_feats_ptr[samp_i];
    int32_t num_s_timesteps = cuda_multi_num_s_timesteps[samp_i];

    int max_last_layer = num_k_timesteps-1 + max_num_s_timesteps-1;

    float *k_feat_weights_intm_ptr,
    *k_added_feats_intm_ptr, *s_added_feats_intm_ptr;
    set_intm_ptrs(
        k_feat_weights_ptr,
        k_added_feats_ptr, s_added_feats_ptr,
        num_kernels, 
        num_k_timesteps, 
        num_added_feats,
        num_samples,                            
        kid,
        num_s_timesteps, 
        k_feat_weights_intm_ptr,
        k_added_feats_intm_ptr, s_added_feats_intm_ptr
    );

    auto calced_dist_ptr = cuda_multi_calced_dist_ptr[samp_i];
 

    int last_layer = num_k_timesteps-1 + num_s_timesteps-1;
    int kt_rem = blockIdx.y * blockDim.y + threadIdx.y;



#ifndef MM_TRANSITION
    // batch_size * num_kernels * (max_last_layer+1) * num_k_timesteps * sizeof(float)
    float *prv = nullptr, *cur = &multi_tmp3[calcIndex(batch_size, samp_i%batch_size, num_kernels, kid, max_last_layer+1, 0, num_k_timesteps, 0)];
    for(int kt = kt_rem; kt < num_k_timesteps; kt += THR_CNT) {
        cur[kt] = (kt == 0 ? 0.0 : 1e38);
    }


    for(int layer = 1; layer <= last_layer; layer++) {
        prv = cur;
        cur = &cur[num_k_timesteps];

        int cur_range_l = max(0, layer-(num_s_timesteps-1)), cur_range_r = min(num_k_timesteps-1, layer);
        
        __syncwarp();
        for(int kt = cur_range_l + kt_rem; kt <= cur_range_r; kt += THR_CNT) {
            int st = layer - kt;
            cur[kt] = min(  st > 0 ? prv[kt] : 1e38, 
                            kt > 0 ? prv[kt-1] : 1e38) +
                      calced_dist_ptr[calcIndex(num_kernels, kid, num_k_timesteps, kt, num_s_timesteps, st)]; //         // kid, kt, num_s_timesteps x 
        }
    }
    __syncwarp();


    #define MARK_PATH_L(layer, kt) attribute_var_intm(\
            k_added_feats_intm_ptr, s_added_feats_intm_ptr,\
            num_kernels, \
            num_k_timesteps, \
            num_added_feats,\
            num_samples,\
            samp_i, kid, (kt), ((layer)-(kt)),\
            k_feat_slope_blob, k_feat_intercept_blob,\
            num_s_timesteps,\
            blob_offset\
        )

    if(kt_rem == THR_CNT-1) {
        int64_t blob_offset = calcIndex(num_samples, samp_i, num_kernels, kid, num_k_timesteps, num_k_timesteps-1, num_added_feats/*+1*/, 0);
        for(int layer = last_layer, kt = num_k_timesteps-1; layer>=1; layer--) {
            MARK_PATH_L(layer, kt);
            int st = layer - kt;
            if((kt > 0 ? prv[kt-1] : 1e38) < (st > 0 ? prv[kt] : 1e38)) {
                kt--;
                blob_offset -= num_added_feats/*+1*/;
            }
            cur = prv;
            prv = &prv[-num_k_timesteps];
        }
        MARK_PATH_L(0, 0);
    }
    #undef MARK_PATH_L
#else 
    // batch_size * num_kernels * (max_last_layer+1) * num_k_timesteps * sizeof(float)
    float *cur = &multi_tmp3[calcIndex(batch_size, samp_i%batch_size, num_kernels, kid, max_last_layer+1, 0, num_k_timesteps, 0)];
    float *prv1 = nullptr; // &cur[2*num_k_timesteps];
    float *prv2 = nullptr; 
    for(int kt = kt_rem; kt < num_k_timesteps; kt += THR_CNT) {
        cur[kt] = (kt == 0 ? 0.0 : 1e38);
        // prv1[kt] = 1e38;
    }

    for(int layer = 1; layer <= last_layer; layer++) {
        prv2 = prv1;
        prv1 = cur;
        cur = &cur[num_k_timesteps];

        int cur_range_l = max(0, layer-(num_s_timesteps-1)), cur_range_r = min(num_k_timesteps-1, layer);
        
        __syncwarp();
        for(int kt = cur_range_l + kt_rem; kt <= cur_range_r; kt += THR_CNT) {
            int st = layer - kt;
            cur[kt] = min(  min(
                            st > 0 ? prv1[kt] : 1e38, 
                            kt > 0 ? prv1[kt-1] : 1e38),
                            st > 0 && kt > 0 ? prv2[kt-1] : 1e38
                            ) +
                      calced_dist_ptr[calcIndex(num_kernels, kid, num_k_timesteps, kt, num_s_timesteps, st)]; //         // kid, kt, num_s_timesteps x 
        }
    }
    __syncwarp();


    #define MARK_PATH_K(kt, st) attribute_var_intm(\
            k_added_feats_intm_ptr, s_added_feats_intm_ptr,\
            num_kernels, \
            num_k_timesteps, \
            num_added_feats,\
            num_samples,\
            samp_i, kid, (kt), (st),\
            k_feat_slope_blob, k_feat_intercept_blob,\
            num_s_timesteps,\
            blob_offset\
        )

    if(kt_rem == THR_CNT-1) {
        int64_t blob_offset = calcIndex(num_samples, samp_i, num_kernels, kid, num_k_timesteps, num_k_timesteps-1, num_added_feats/*+1*/, 0);
        for(int kt = num_k_timesteps-1, st = num_s_timesteps-1; kt+st>=1; ) {
            MARK_PATH_K(kt, st);
            float ktm = (kt > 0 ? prv1[kt-1] : 1e38);
            float stm = (st > 0 ? prv1[kt] : 1e38);
            float ktstm = (st > 0 && kt > 0 ? prv2[kt-1] : 1e38);


            if(ktm < stm && ktm < ktstm) {
                kt--;
                blob_offset -= num_added_feats/*+1*/;
                
                cur = prv1;
                prv1 = prv2;
                prv2 = &prv2[-num_k_timesteps];
            } else if(stm < ktstm) {
                st--;
                
                cur = prv1;
                prv1 = prv2;
                prv2 = &prv2[-num_k_timesteps];
            } else {
                kt--; st--;
                blob_offset -= num_added_feats/*+1*/;
                
                cur = prv2;
                prv1 = &prv2[-num_k_timesteps];
                prv2 = &prv2[-2*num_k_timesteps];
            }
        }
        MARK_PATH_K(0, 0);
    }
    #undef MARK_PATH_K
#endif 
}

template<int batch_size>
void run_bt(
    #ifdef FIXED_PATH
    torch::Tensor _k_added_feats_fixed,
    torch::Tensor _k_feat_weights_fixed,
    #endif 
    torch::Tensor _k_added_feats, 
    torch::Tensor _k_feat_weights,
    vector<torch::Tensor> _multi_s_added_feats, 
    const vector<int64_t> &multi_num_s_timesteps,
    const int64_t& num_kernels, 
    const int64_t& num_k_timesteps, 
    const int64_t& num_added_feats,
    const int64_t& num_samples,
    torch::Tensor ret1, 
    torch::Tensor ret2,
    const int& feat_batch_size
) {


    float* k_added_feats_ptr = (float*)_k_added_feats.data_ptr();
    float* k_feat_weights_ptr = (float*)_k_feat_weights.data_ptr();
    vector<float*> multi_s_added_feats_ptr(num_samples);
    for(int i=0; i<num_samples; i++) multi_s_added_feats_ptr[i] = (float*)_multi_s_added_feats[i].data_ptr();


    // blob dims: {num_samples, num_kernels, num_k_timesteps, num_added_feats}
    float* k_feat_slope_blob = (float*)ret1.data_ptr();
    float* k_feat_intercept_blob = (float*)ret2.data_ptr();



    int64_t max_num_s_timesteps = *max_element(multi_num_s_timesteps.begin(), multi_num_s_timesteps.end());

    int64_t *cuda_multi_num_s_timesteps; 
    cudaMalloc(&cuda_multi_num_s_timesteps, multi_num_s_timesteps.size() * sizeof(int64_t));
    cudaMemcpy(cuda_multi_num_s_timesteps, multi_num_s_timesteps.data(), multi_num_s_timesteps.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    float** cuda_multi_s_added_feats_ptr;
    cudaMalloc(&cuda_multi_s_added_feats_ptr, multi_s_added_feats_ptr.size() * sizeof(float*));
    cudaMemcpy(cuda_multi_s_added_feats_ptr, multi_s_added_feats_ptr.data(), multi_s_added_feats_ptr.size() * sizeof(float*), cudaMemcpyHostToDevice);



    int max_last_layer = num_k_timesteps-1 + max_num_s_timesteps-1;    

    constexpr int THR_CNT = 32;

    float *multi_tmp3; cudaMalloc(&multi_tmp3, batch_size * num_kernels * (max_last_layer+1) * num_k_timesteps * sizeof(float));


    float** cuda_multi_calced_dist_ptr;
    cudaMalloc(&cuda_multi_calced_dist_ptr, num_samples * sizeof(float*));



    for(int64_t samp_i_offset = 0; samp_i_offset < num_samples; samp_i_offset += batch_size) {
        int64_t cur_size = min((int64_t)batch_size, (int64_t)(num_samples - samp_i_offset));
        vector<torch::Tensor> multi_calced_dist_offset;
        vector<float*> multi_calced_dist_ptr_offset;
        for(int64_t samp_i = samp_i_offset; samp_i < samp_i_offset+cur_size; samp_i++) {
            auto calced_dist = torch::zeros({_k_added_feats.size(0), _k_added_feats.size(1), _multi_s_added_feats[samp_i].size(0)}, torch::device(torch::kCUDA).dtype(torch::kFloat32));


            // constexpr int feat_batch_size = 128;
            for(int fl = 0; fl < _multi_s_added_feats[samp_i].size(-1); fl += feat_batch_size) {
                int fr = min(fl + feat_batch_size, (int)_multi_s_added_feats[samp_i].size(-1));
                #ifdef FIXED_PATH
                #define _kfw _k_feat_weights_fixed
                #define _kaf _k_added_feats_fixed
                #else 
                #define _kfw _k_feat_weights
                #define _kaf _k_added_feats
                #endif 
                auto lhs = _kfw.slice(-1, fl, fr).unsqueeze(2).expand({-1, -1, _multi_s_added_feats[samp_i].size(0), -1}) * 
                            torch::abs(
                                _kaf.slice(-1, fl, fr).unsqueeze(2).expand({-1, -1, _multi_s_added_feats[samp_i].size(0), -1}) - 
                                _multi_s_added_feats[samp_i].slice(-1, fl, fr).unsqueeze(0).unsqueeze(0).expand({_kaf.size(0), _kaf.size(1), -1, -1}));
                calced_dist += lhs.sum(-1); 
            }
            // kid, kt, num_s_timesteps x num_remaining_feats
            multi_calced_dist_offset.push_back(calced_dist);
            multi_calced_dist_ptr_offset.push_back((float*)multi_calced_dist_offset.back().data_ptr());
        }

        cudaMemcpy(&cuda_multi_calced_dist_ptr[samp_i_offset], multi_calced_dist_ptr_offset.data(), multi_calced_dist_ptr_offset.size() * sizeof(float*), cudaMemcpyHostToDevice);

        // Number of thread blocks in grid
        dim3 dimBlock(1, THR_CNT, 1);
        dim3 dimGrid((min(batch_size, (int)(num_samples-samp_i_offset)) + dimBlock.x - 1) / dimBlock.x,\
                        1,\
                        (num_kernels + dimBlock.z - 1) / dimBlock.z);

        // Execute the kernel
        per_samp_i_kid_kt<THR_CNT, batch_size><<<dimGrid, dimBlock>>>(
            k_feat_weights_ptr,
            k_added_feats_ptr, cuda_multi_s_added_feats_ptr, 
            cuda_multi_calced_dist_ptr,
            k_feat_slope_blob,
            k_feat_intercept_blob,
            num_kernels, 
            num_k_timesteps, 
            num_added_feats,
            num_samples,
            cuda_multi_num_s_timesteps, 
            multi_tmp3,
            max_num_s_timesteps,
            samp_i_offset
        );
        cudaDeviceSynchronize();
    }


    cudaFree(multi_tmp3);
    cudaFree(cuda_multi_num_s_timesteps);
    cudaFree(cuda_multi_s_added_feats_ptr);

    cudaFree(cuda_multi_calced_dist_ptr);


}


void run(
    #ifdef FIXED_PATH
    torch::Tensor _k_added_feats_fixed, 
    torch::Tensor _k_feat_weights_fixed,
    #endif
    torch::Tensor _k_added_feats, 
    torch::Tensor _k_feat_weights,
    vector<torch::Tensor> _multi_s_added_feats, 
    const vector<int64_t> &multi_num_s_timesteps,
    const int64_t& num_kernels, 
    const int64_t& num_k_timesteps, 
    const int64_t& num_added_feats,
    const int64_t& num_samples,
    torch::Tensor ret1, 
    torch::Tensor ret2,
    const int& feat_batch_size
) {
    if(num_samples <= 1) {
        run_bt<1>(
            #ifdef FIXED_PATH
            _k_added_feats_fixed,
            _k_feat_weights_fixed,
            #endif
            _k_added_feats, _k_feat_weights, _multi_s_added_feats, multi_num_s_timesteps, num_kernels, num_k_timesteps, num_added_feats, num_samples, ret1, ret2, feat_batch_size);
    } else if(num_samples <= 2) {
        run_bt<2>(
            #ifdef FIXED_PATH
            _k_added_feats_fixed,
            _k_feat_weights_fixed,
            #endif
            _k_added_feats, _k_feat_weights, _multi_s_added_feats, multi_num_s_timesteps, num_kernels, num_k_timesteps, num_added_feats, num_samples, ret1, ret2, feat_batch_size);
    } else if(num_samples <= 4) {
        run_bt<4>(
            #ifdef FIXED_PATH
            _k_added_feats_fixed,
            _k_feat_weights_fixed,
            #endif
            _k_added_feats, _k_feat_weights, _multi_s_added_feats, multi_num_s_timesteps, num_kernels, num_k_timesteps, num_added_feats, num_samples, ret1, ret2, feat_batch_size);
    } else if(num_samples <= 8) {
        run_bt<8>(
            #ifdef FIXED_PATH
            _k_added_feats_fixed,
            _k_feat_weights_fixed,
            #endif
            _k_added_feats, _k_feat_weights, _multi_s_added_feats, multi_num_s_timesteps, num_kernels, num_k_timesteps, num_added_feats, num_samples, ret1, ret2, feat_batch_size);
    } else {
        run_bt<16>(
            #ifdef FIXED_PATH
            _k_added_feats_fixed,
            _k_feat_weights_fixed,
            #endif
            _k_added_feats, _k_feat_weights, _multi_s_added_feats, multi_num_s_timesteps, num_kernels, num_k_timesteps, num_added_feats, num_samples, ret1, ret2, feat_batch_size);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "a func");
}
