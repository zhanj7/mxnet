#include "./assign_sample-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
  namespace op {
    template<>
    Operator* CreateOp<cpu>(AssignSampleParam param) {
      return new AssignSampleOp<cpu>(param);
    }

    Operator* AssignSampleProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(AssignSampleParam);

    MXNET_REGISTER_OP_PROPERTY(AssignSample, AssignSampleProp)
    .describe("Assign negative samples to form a fixed size training sample set."
	      "Only pass gradients to samples used in training")
    .set_return_type("Symbol[]")
    .add_argument("clsdata", "Symbol", "Input cls data.")    
    .add_argument("regdata", "Symbol", "Input reg data.")
    .add_argument("label", "Symbol", "Input cls label.")
    .add_arguments(AssignSampleParam::__FIELDS__());
  }
}
