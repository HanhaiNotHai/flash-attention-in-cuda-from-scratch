"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float *a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] *= scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float *a, int n) {
    // TODO: replace each a[i] with expf(a[i])

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] = expf(a[i]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float *matrix, float *out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].

    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float max_val = -INFINITY;

        for (int c = 0; c < cols; c++) {
            max_val = fmaxf(max_val, matrix[r * cols + c]);
        }

        out[r] = max_val;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float *matrix, float *out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r

    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float sum = 0.0f;

        for (int c = 0; c < cols; c++) {
            sum += matrix[r * cols + c];
        }

        out[r] = sum;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float *a, const float *b, int n) {
    // TODO: return the dot product of a and b

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float *a, const float *b, float *c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        float sum = 0.0f;

        for (int l = 0; l < k; l++) {
            sum += a[i * k + l] * b[l * n + j];
        }

        c[i * n + j] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float *in, float *out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float *q, const float *k, float *scores, int seq_len, int head_dim) {
    // TODO: compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        scores[i * seq_len + j] = dot_product(q + i * head_dim, k + j * head_dim, head_dim) / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float *matrix, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax in place

    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float *row_data = matrix + row * cols;

    // 1. Find row maximum
    float local_max = -INFINITY;

    for (int c = tid; c < cols; c += blockDim.x) {
        float val = row_data[c];
        if (val > local_max) {
            local_max = val;
        }
    }

    shared[tid] = local_max;
    __syncthreads();

    // reduction max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && shared[tid + stride] > shared[tid]) {
            shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    float row_max = shared[0];

    // 2. Compute exp(x-max) and sum
    float local_sum = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        float e = expf(row_data[c] - row_max);
        row_data[c] = e;
        local_sum += e;
    }

    shared[tid] = local_sum;
    __syncthreads();

    // reduction sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = shared[0];

    // 3. Normalize
    for (int c = tid; c < cols; c += blockDim.x) {
        row_data[c] /= row_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float *p, const float *v, float *out, int seq_len, int head_dim) {
    // TODO: compute out[i, d] = sum_j p[i, j] * v[j, d]

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;

        for (int j = 0; j < seq_len; j++) {
            sum += p[i * seq_len + j] * v[j * head_dim + d];
        }

        out[i * head_dim + d] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float *d_q, const float *d_k, const float *d_v, float *d_out, int seq_len, int head_dim) {
    // TODO: allocate scratch, launch qk_scores -> softmax_rows -> pv_matmul, free scratch

    // allocate score matrix
    float *d_scores;

    size_t score_bytes = seq_len * seq_len * sizeof(float);

    cudaMalloc(&d_scores, score_bytes);

    // 1. QK^T / sqrt(d)
    dim3 block_qk(16, 16);

    dim3 grid_qk((seq_len + block_qk.x - 1) / block_qk.x, (seq_len + block_qk.y - 1) / block_qk.y);

    qk_scores<<<grid_qk, block_qk>>>(d_q, d_k, d_scores, seq_len, head_dim);

    // 2. Row-wise softmax
    int softmax_threads = 256;

    size_t shared_bytes = softmax_threads * sizeof(float);

    softmax_rows<<<seq_len, softmax_threads, shared_bytes>>>(d_scores, seq_len, seq_len);

    // 3. P * V
    dim3 block_pv(16, 16);

    dim3 grid_pv((head_dim + block_pv.x - 1) / block_pv.x, (seq_len + block_pv.y - 1) / block_pv.y);

    pv_matmul<<<grid_pv, block_pv>>>(d_scores, d_v, d_out, seq_len, head_dim);

    // free scratch memory
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val

    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

