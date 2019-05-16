#include "./tripletloss-inl.h"

namespace mshadow {

namespace {
// workspace variables
enum TripletLossTempSpaceType {kDistance};
}

template <typename DType>
inline void TripletLossForward(const Tensor<cpu, 2, DType> &distance,
		                       const Tensor<cpu, 1, DType> &label,
		                       Tensor<cpu, 1, DType> &out,
		                       const int n, const float margin, const float epsilon) {

	LOG(FATAL) << "C OP Not finished";

	for(int a=0; a<n; a++) {
		DType min_ndist = 3.0;
		DType mean_pdist = 0.0;
		int npos = 0;

		for(int j=0; j<n; j++) {
			if(j == a) continue;

			DType dist = distance[a][j];

			if(label[j] == label[a]) {
				mean_pdist += dist;
				npos += 1;
			} else {
				if(dist < min_ndist)
					min_ndist = dist;
			}
		}

		mean_pdist /= npos;
		out[a] = mean_pdist + epsilon * MAX(mean_pdist - min_ndist + margin, static_cast<DType>(0.0));

	}

}

template <typename DType>
inline void TripletLossBackward(Tensor<cpu, 3, DType> &workspace,
		                        const Tensor<cpu, 2, DType> &data,
		                        const Tensor<cpu, 1, DType> &label,
		                        Tensor<cpu, 2, DType> &in_grad,
		                        const int n, const int k, const float margin, const float epsilon) {

	using namespace mshadow::expr;

	LOG(FATAL) << "C OP Not finished";

	// workspace[kDistance] = dot(data, data.T());

	workspace[kDistance][0] = sumall_except_dim<0>(data * data);
	workspace[kDistance] = repmat(workspace[kDistance][0], n);
	workspace[kDistance] += broadcast<0>(workspace[kDistance][0], workspace[kDistance].shape_);
	// workspace[kDistance] *= 0.5;
	// workspace[kDistance] -= dot(data, data.T());

	std::cout << workspace[kDistance][0][0] << " " << workspace[kDistance][0][1] << std::endl;
	std::cout << workspace[kDistance][1][0] << " " << workspace[kDistance][1][1] << std::endl;

	in_grad = 0.0;

	for(int a=0; a<n; a++) {
		int npos = 0;
		int min_nid = 0;
		DType min_ndist = 3.0;
		DType mean_pdist = 0.0;

		for(int j=0; j<n; j++)
			if(j != a && label[j] == label[a])
				npos++;

		for(int j=0; j<n; j++) {
			if(j == a) continue;

			DType dist = workspace[kDistance][a][j];

			if(label[j] == label[a]) {
				in_grad[a] += (data[a] - data[j]);
				in_grad[j] += (data[j] - data[a]);
				mean_pdist += dist;
			} else {
				if(dist < min_ndist) {
					min_ndist = dist;
					min_nid = j;
				}
			}
		}

		in_grad[a] /= npos;
		mean_pdist /= npos;
		for(int j=0; j<n; j++) {
			if(j != a && label[j] == label[a]) {
				in_grad[j] /= npos;
			}
		}

		DType distdiff = mean_pdist - min_ndist + margin;
		if(distdiff > 0) {
			// in_grad[min_nid] += epsilon * (data[a] - data[min_nid]);
			in_grad[min_nid] /= epsilon;
			in_grad[min_nid] += (data[a] - data[min_nid]);
			in_grad[min_nid] *= epsilon;

			// in_grad[a] += epsilon * data[min_nid];
			in_grad[a] /= epsilon;
			in_grad[a] += data[min_nid];
			in_grad[a] *= epsilon;

			for(int j=0; j<n; j++) {
				if(j != a && label[j] == label[a]) {

					float scale_factor = epsilon / npos;

					// in_grad[a] -= epsilon * (data[j] / npos);
					in_grad[a] /= scale_factor;
					in_grad[a] -= data[j];
					in_grad[a] *= scale_factor;

					// in_grad[j] += epsilon * (data[j] - data[a]) / npos;
					in_grad[j] /= scale_factor;
					in_grad[j] += data[j] - data[a];
					in_grad[j] *= scale_factor;

				}
			}
		}

	}
}

} // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(TripletLossParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {op = new TripletLossOp<cpu, DType>(param);})

  return op;
}

// Do bind_dispatch comes from operator_common.h
Operator *TripletLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(TripletLossParam);

MXNET_REGISTER_OP_PROPERTY(TripletLoss, TripletLossProp)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(TripletLossParam::__FIELDS__());

} // namespace op
} // namespace mxnet
