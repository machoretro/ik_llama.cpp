//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "mmvq.cuh"
#include "quantize.cuh"
#include "iqk_mmvq.cuh"
#include "vecdotq.cuh"

#include <cstdint>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0   : return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1   : return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0   : return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1   : return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q6_0   : return vec_dot_q6_0_q8_1;
        case GGML_TYPE_Q8_0   : return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K   : return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K   : return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K   : return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K   : return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K   : return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS : return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S  : return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S  : return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M  : return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL : return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_MXFP4  : return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_IQ4_XS : return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S  : return vec_dot_iq3_s_q8_1;
        default               : return nullptr;
    }
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0    : return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1    : return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0    : return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1    : return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q6_0    : return VDR_Q6_0_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0    : return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K    : return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K    : return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K    : return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K    : return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K    : return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS : return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS  : return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S   : return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS : return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S   : return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL  : return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4   : return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS  : return VDR_IQ4_XS_Q8_1_MMVQ;
        default                : return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA2) || defined(RDNA3) || defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst,  mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst>
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols_x, const int nchannels_y, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    // The MUL_MAT_ID code path with ids != nullptr is only implemetned for ncols_dst == 1.
    const int channel_dst = blockIdx.y;
    const int channel_x   = ncols_dst == 1 && ids ? ids[channel_dst]          : channel_dst / channel_ratio;
    const int channel_y   = ncols_dst == 1 && ids ? channel_dst % nchannels_y : channel_dst;
    const int sample_dst  = blockIdx.z;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + int(threadIdx.x) < stride_col_dst)) {
            dst[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_y, const int nsamples_y,
        const int warp_size, const mmvq_parameter_table_id table_id) {
    const int64_t nblocks = (nrows_x + calc_rows_per_block(ncols_dst, table_id) - 1) / calc_rows_per_block(ncols_dst, table_id);
    const dim3 block_nums(nblocks, nchannels_y, nsamples_y);
    const dim3 block_dims(warp_size, calc_nwarps(ncols_dst, table_id), 1);
    return {block_nums, block_dims};
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const int channel_ratio = nchannels_dst / nchannels_x;
    const int sample_ratio  = nsamples_dst  / nsamples_x;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    GGML_ASSERT(!ids || ncols_dst == 1);
    switch (ncols_dst) {
        case 1:
        {
            constexpr int c_ncols_dst = 1;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 2:
        {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 3:
        {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 4:
        {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 5:
        {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 6:
        {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 7:
        {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        case 8:
        {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 == 1); // Implementation is only correct for  batch size 1.

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,                 stream);
}

template <ggml_type type, int ncols_y, int nwarps>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const char * __restrict__ ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0) {
    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        // We clear the buffer via cudaMemset instead
//#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
//        constexpr int rows_per_cuda_block = 1;
//#else
//        constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
//#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)
//        const int row0 = rows_per_cuda_block*blockIdx.x;
//        if (threadIdx.y == 0) {
//            dst = (float *)cdst;
//            for (int j = 0; j < ncols_y; ++j) {
//                if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
//                    dst[j*nrows_dst + row0 + threadIdx.x] = 0;
//                }
//            }
//        }
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    mul_mat_vec_q<type, ncols_y, nwarps>(cx, cy, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

template <ggml_type type, int nwarps>
static void mul_mat_vec_q_cuda_T(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < CC_RDNA2 ?
        ncols_y < 4 ? 1 : 2 : 1;

    //if (ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
    //    switch(ncols_y) {
    //        case 1:
    //            nwarps = 4;
    //            rows_per_cuda_block = 1;
    //            break;
    //        case 2:
    //        case 3:
    //        case 4:
    //            nwarps = 4;
    //            rows_per_cuda_block = 2;
    //            break;
    //        case 5:
    //        case 6:
    //        case 7:
    //        case 8:
    //            nwarps = 2;
    //            rows_per_cuda_block = 2;
    //            break;
    //        default:
    //            GGML_ABORT("fatal error");
    //            break;
    //    }
    //}
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<type, 1, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 2:
            mul_mat_vec_q<type, 2, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 3:
            mul_mat_vec_q<type, 3, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 4:
            mul_mat_vec_q<type, 4, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 5:
            mul_mat_vec_q<type, 5, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 6:
            mul_mat_vec_q<type, 6, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 7:
            mul_mat_vec_q<type, 7, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 8:
            mul_mat_vec_q<type, 8, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {
    int nwarps = 1;
    int id = ggml_cuda_get_device();
    if (ne2 < 2 && ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        nwarps = ncols_y <= 4 ? 4 : 2;
    }
    switch (nwarps) {
        case 1:
            mul_mat_vec_q_cuda_T<type, 1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case 2:
            mul_mat_vec_q_cuda_T<type, 2>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            mul_mat_vec_q_cuda_T<type, 4>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
    }
}

static void mul_mat_vec_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q6_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q6_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q8_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q2_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q3_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q6_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_XXS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_XS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_S>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_XXS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ1_S>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ1_M>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ4_NL>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_mxfp4_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_MXFP4>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ4_XS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_S>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void ggml_cuda_op_mul_mat_vec_q_impl(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0,
        const char * src0_dd_i, const char * src1_ddq_i, float * dst_dd_i, const char * ids_data,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q6_0:
            mul_mat_vec_q6_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_iq2_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_iq2_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_iq2_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_iq3_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_iq1_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_iq1_m_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_BN:
            mul_mat_vec_iq1_bn_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_BN:
            mul_mat_vec_iq2_bn_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_iq4_nl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_mxfp4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_iq4_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_vec_iq2_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_vec_iq3_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_vec_iq2_kl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_vec_iq3_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_vec_iq4_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_vec_iq4_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_vec_iq4_kss_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_vec_iq1_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_vec_iq2_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_vec_iq3_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_vec_iq4_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_vec_iq2_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_vec_iq5_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_vec_iq6_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_iq3_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_vec_iq2_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_vec_iq3_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_vec_iq4_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_vec_iq4_ks_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_vec_iq5_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_vec_iq5_ks_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_vec_iq1_s_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_M_R4:
            mul_mat_vec_iq1_m_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

}

void ggml_cuda_op_mul_mat_vec_q_3D(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src0->ne[2] == src1->ne[2] && src0->ne[2] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    const int64_t src1_row_size = ggml_row_size(GGML_TYPE_Q8_1, src1_padded_row_size);

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1_row_size, dst->nb[2], 0,
        src0_dd_i, src1_ddq_i, dst_dd_i, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stream);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}

void ggml_cuda_op_mul_mat_vec_q_id(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1);
    GGML_ASSERT(ids->ne[0] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1->nb[2], dst->nb[2], ids->nb[0],
        src0_dd_i, src1_ddq_i, dst_dd_i, (const char *)ids->data,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_mmvq_type_supported(ggml_type src0_type) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return true;
        default:
            return false;
    }
}
