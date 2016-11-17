/*!
 * Copyright (c) 2015 by Contributors
 * \file scale.cc
 * \brief scale operator
*/
#include "./scale-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ScaleParam param) {
  return new ScaleOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ScaleProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ScaleParam);

MXNET_REGISTER_OP_PROPERTY(Scale, ScaleProp)
.describe("Scale input then add a bias.")
.add_argument("data", "Symbol", "Input data to the ScaleOp.")
.add_argument("gamma", "Symbol", "Scale Parameter.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(ScaleParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
