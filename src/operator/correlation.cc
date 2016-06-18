/*!
 * Copyright (c) 2015 by Contributors
 * \file correlation.cc
 * \brief
 * \author Xu Dong
*/

#include "./correlation-inl.h"

namespace mshadow {
  
template<typename Dtype>
inline void CorrelationForward( const Tensor<cpu, 4, Dtype> &out,
                                const Tensor<cpu, 4, Dtype> &data1,
                                const Tensor<cpu, 4, Dtype> &data2,
                                const Tensor<cpu, 4, Dtype> &tmp1,
                                const Tensor<cpu, 4, Dtype> &tmp2,
                                int top_channels_,int top_height_,int top_width_,int pad_size_,bool is_multiply,
                                int max_displacement_,int kernel_size_,int neighborhood_grid_radius_,int neighborhood_grid_width_,
                                int  kernel_radius_,int stride1_,int stride2_
                           ) {
  return  ; 
}

template<typename Dtype>
inline void CorrelationBackward(const Tensor<cpu, 4, Dtype> &out_grad,
                            const Tensor<cpu, 4, Dtype> &in_grad1,
                            const Tensor<cpu, 4, Dtype> &in_grad2,
                            const Tensor<cpu, 4, Dtype> &tmp1, 
                            const Tensor<cpu, 4, Dtype> &tmp2,
                            int top_channels_,int top_height_,int top_width_,int pad_size_,bool is_multiply,
                            int max_displacement_,int kernel_size_,int neighborhood_grid_radius_,int neighborhood_grid_width_,
                            int  kernel_radius_,int stride1_,int stride2_,int num, int channels,int height, int width
                            ) {
                              
 return ; 
}
}  // namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(CorrelationParam param) {
  return new CorrelationOp<cpu>(param);
}

Operator* CorrelationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CorrelationParam);

MXNET_REGISTER_OP_PROPERTY(Correlation, CorrelationProp)
.describe("Apply correlation to inputs")
.add_argument("data1", "Symbol", "Input data to the correlation.")
.add_argument("data2", "Symbol", "Input data to the correlation.")
.add_arguments(CorrelationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
