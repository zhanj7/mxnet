/*!
 * Copyright (c) 2015 by Contributors
 * \file assign_sample.cu
 * \brief
 * \author Xuan Lin
*/
#include "./assign_sample-inl.h"

namespace mxnet {
namespace op {
template<>
  Operator *CreateOp<gpu>(AssignSampleParam param) {
    return new AssignSampleOp<gpu>(param);
  }
}  // namespace op
}  // namespace mxnet
