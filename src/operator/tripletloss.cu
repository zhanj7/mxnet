#include "./tripletloss-inl.h"

#define N 256 // batch_size n should not be greater than N 256

namespace mshadow {

namespace {
// workspace variables
enum TripletLossTempSpaceType {kDistance, kGradWeight};
}

namespace cuda {

template <typename DType>
__global__ void TripletLossForwardKernel(const Tensor<gpu, 2, DType> distance,
		                                 const Tensor<gpu, 1, DType> label,
		                                 Tensor<gpu, 1, DType> out,
		                                 const int n, const float margin, const float epsilon) {

	const int tid = threadIdx.x;
	const int a = blockIdx.x;

	if(a < n) {
		const DType label_a = label[a];
		__shared__ int min_nid[N];
		__shared__ DType min_ndist[N];
        __shared__ DType npos[N];
        __shared__ DType mean_pdist[N];

		if(tid < n) {
			// load data into shared memory
			min_nid[tid] = tid;
			min_ndist[tid] = distance[a][tid];
			if(label[tid] == label_a) {
				// min_ndist[tid] = 3.0;
				npos[tid] = 1.0;
				mean_pdist[tid] = distance[a][tid];
			}
			else {
				// min_ndist[tid] = distance[a][tid];
				npos[tid] = 0.0;
				mean_pdist[tid] = 0.0;
			}
			__syncthreads();

			// reduction
			for(int step=1; step<n; step<<=1) {
				if( 2*tid*step + step < n) {
					// min_ndist[2*tid*step] = MIN(min_ndist[2*tid*step], min_ndist[2*tid*step + step]);
					if(min_ndist[2*tid*step] > min_ndist[2*tid*step + step] || label[min_nid[2*tid*step]] == label_a) {
						if(label[min_nid[2*tid*step + step]] != label_a) {
							min_ndist[2*tid*step] = min_ndist[2*tid*step + step];
							min_nid[2*tid*step] = min_nid[2*tid*step + step];
						}
					}
					npos[2*tid*step] = npos[2*tid*step] + npos[2*tid*step + step];
					mean_pdist[2*tid*step] = mean_pdist[2*tid*step] + mean_pdist[2*tid*step + step];
				}
				__syncthreads();
			}

			// calculate loss
			if(tid == 0) {
				out[a] = mean_pdist[0]/(npos[0]-1) + epsilon * MAX(mean_pdist[0]/(npos[0]-1) - min_ndist[0] + margin, static_cast<DType>(0.0));
			}
		}
	}
}

template <typename DType>
__global__ void TripletLossBackwardKernel(const Tensor<gpu, 2, DType> distance,
		                                  const Tensor<gpu, 2, DType> data,
		                                  const Tensor<gpu, 1, DType> label,
		                                  Tensor<gpu, 2, DType> grad_w,
		                                  const int n, const float margin, const float epsilon) {

	const int j = threadIdx.x;
	const int a = blockIdx.x;

	if(a < n) {
		const DType label_a = label[a];
		__shared__ int min_nid[N];
		__shared__ DType min_ndist[N];
		__shared__ DType npos[N];
		__shared__ DType mean_pdist[N];

		if(j < n) {

			// load data into shared memory
			min_nid[j] = j;
			min_ndist[j] = distance[a][j];
			if(label[j] == label_a) {
				// min_ndist[j] = 3.0; // bad code here
				npos[j] = 1.0;
				mean_pdist[j] = distance[a][j];
			} else {
				// min_ndist[j] = distance[a][j];
				npos[j] = 0.0;
				mean_pdist[j] = 0.0;
			}
			__syncthreads();

			// reduction
			for(int step=1; step<n; step<<=1) {
				if( 2*j*step + step < n ) {
					// min_ndist[2*j*step] = MIN(min_ndist[2*j*step], min_ndist[2*j*step + step]);
					// min_nid[2*j*step] = ARGMIN(min_ndist[2*j*step], min_ndist[2*j*step + step]);
					if(min_ndist[2*j*step] > min_ndist[2*j*step + step] || label[min_nid[2*j*step]] == label_a) {
						if(label[min_nid[2*j*step + step]] != label_a) {
							min_ndist[2*j*step] = min_ndist[2*j*step + step];
							min_nid[2*j*step] = min_nid[2*j*step + step];
						}
					}
					npos[2*j*step] = npos[2*j*step] + npos[2*j*step + step];
					mean_pdist[2*j*step] = mean_pdist[2*j*step] + mean_pdist[2*j*step + step];
				}
				__syncthreads();
			}

			// calculate npos, mean_pdist, min_ndist, min_nid
			int npos_scalar = npos[0] - 1;
			DType mean_pdist_scalar = mean_pdist[0] / npos_scalar;
			DType min_ndist_scalar = min_ndist[0];
			int min_nid_scalar = min_nid[0];
			DType distdiff = mean_pdist_scalar - min_ndist_scalar + margin;

			// atomic add
			if(label[j] == label_a && j != a) {
				// grad_a
				atomicAdd(&grad_w[a][a], 1.0/npos_scalar);
				atomicAdd(&grad_w[a][j], -1.0/npos_scalar);
                // grad_p
				atomicAdd(&grad_w[j][j], 1.0/npos_scalar);
				atomicAdd(&grad_w[j][a], -1.0/npos_scalar);
			}

			if(distdiff > 0) {
				if(label[j] == label_a && j != a) {
					// grad_a
					atomicAdd(&grad_w[a][min_nid_scalar], epsilon * 1.0/npos_scalar);
					atomicAdd(&grad_w[a][j], epsilon * (-1.0/npos_scalar));
					// grad_p
					atomicAdd(&grad_w[j][j], epsilon * 1.0/npos_scalar);
					atomicAdd(&grad_w[j][a], epsilon * (-1.0/npos_scalar));
					// grad_n
					atomicAdd(&grad_w[min_nid_scalar][a], epsilon * 1.0/npos_scalar);
					atomicAdd(&grad_w[min_nid_scalar][min_nid_scalar], epsilon * (-1.0/npos_scalar));
				}
			}

		}

	}
}

} // namespace cuda

template <typename DType>
inline void TripletLossForward(const Tensor<gpu, 2, DType> &distance,
		                       const Tensor<gpu, 1, DType> &label,
		                       const Tensor<gpu, 1, DType> &out,
		                       const int n, const float margin, const float epsilon) {

	cuda::TripletLossForwardKernel<<<n, n>>>(distance, label, out, n, margin, epsilon);

}

template <typename DType>
inline void TripletLossBackward(Tensor<gpu, 3, DType> &workspace,
		                        const Tensor<gpu, 2, DType> &data,
		                        const Tensor<gpu, 1, DType> &label,
		                        Tensor<gpu, 2, DType> &in_grad,
		                        const int n, const int k, const float margin, const float epsilon) {
	using namespace mshadow::expr;

	workspace[kGradWeight] = 0.0;

	workspace[kDistance][0] = sumall_except_dim<0>(data * data);
	workspace[kDistance] = repmat(workspace[kDistance][0], n);
	workspace[kDistance] += workspace[kDistance].T();
	workspace[kDistance] *= 0.5;
	workspace[kDistance] -= dot(data, data.T());

	cuda::TripletLossBackwardKernel<<<n, n>>>(workspace[kDistance], data, label, workspace[kGradWeight], n, margin, epsilon);
	in_grad = dot(workspace[kGradWeight], data);
}

} // namespace mshadow


namespace mxnet {
namespace op {

// inherit template from header file
template<>
Operator *CreateOp<gpu>(TripletLossParam param, int dtype) {
	Operator *op = NULL;
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {op = new TripletLossOp<gpu, DType>(param); })

	return op;
}

}
}
