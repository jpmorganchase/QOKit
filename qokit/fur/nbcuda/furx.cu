template <int n_q, int q_offset>
__device__ uint2 get_offset(const unsigned int block_idx){
    constexpr unsigned int stride = 1 << q_offset;
    constexpr unsigned int index_mask = stride - 1;
    constexpr unsigned int stride_mask = ~index_mask;
    const unsigned int offset = ((stride_mask & block_idx) << n_q) | (index_mask & block_idx);

    return {offset, stride};
}

__device__ constexpr double2 rot_x(const double a, const double b, double2& va, double2& vb){
    double2 temp = {a*va.x - b*vb.y, a*va.y + b*vb.x};
    vb = {a*vb.x - b*va.y, a*vb.y + b*va.x};
    va = temp;
}

template <int n_q, int q_offset, int state_mask>
__global__ void furx_kernel(double2* x, const double a, const double b) {
    __shared__ double2 data[1 << n_q];
    constexpr unsigned int stride_size = 1 << (n_q-1);

    auto [offset, stride] = get_offset<n_q, q_offset>(blockIdx.x);
    const unsigned int tid = threadIdx.x;              
                
    for(int i = 0; i < 2; ++i){
        const unsigned int idx = (tid+stride_size*i);
        data[idx] = x[offset + idx*stride];
    }

    __syncthreads();

    for(int q = 0; q < n_q; ++q){
        const unsigned int mask1 = (1 << q) - 1;
        const unsigned int mask2 = state_mask - mask1;

        const unsigned int ia = (tid & mask1) | ((tid & mask2) << 1);
        const unsigned int ib = ia | (1 << q);

        rot_x(a,b, data[ia], data[ib]);

        __syncthreads();
    }
    
    for(int i = 0; i < 2; ++i){
        const unsigned int idx = (tid+stride_size*i);
        x[offset + idx*stride] = data[idx];
    }
}

template <int n_q, int q_offset>
__global__ void warp_furx_kernel(double2* x, const double a, const double b) {
    constexpr unsigned int stride_size = 1 << (n_q-1);
    const unsigned int block_idx = blockIdx.x * blockDim.x/stride_size + threadIdx.x/stride_size;
    auto [offset, stride] = get_offset<n_q, q_offset>(block_idx);

    const unsigned int tid = threadIdx.x%stride_size;  
    const unsigned int load_offset = offset + (tid * 2)*stride;   
            
    double2 v[2] = {x[load_offset], x[load_offset + stride]};
    
    rot_x(a, b, v[0], v[1]);

    #pragma unroll
    for(int q = 0; q < n_q-1; ++q){
        const unsigned int warp_stride = 1 << q;
        const bool positive = !(tid & warp_stride);
        const unsigned int lane_idx = positive? tid + warp_stride : tid - warp_stride;

        v[positive].x = __shfl_sync(0xFFFFFFFF, v[positive].x, lane_idx, stride_size);
        v[positive].y = __shfl_sync(0xFFFFFFFF, v[positive].y, lane_idx, stride_size);
        
        rot_x(a, b, v[0], v[1]);
    }
    
    x[offset + tid*stride] = v[0];
    x[offset + (tid + stride_size)*stride] = v[1];
}