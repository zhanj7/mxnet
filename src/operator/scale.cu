/*!
 * Copyright (c) 2015 by Contributors
 * \file scale.cu
 * \brief scale operator
*/
#include "./scale-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ScaleParam param) {
  return new ScaleOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
