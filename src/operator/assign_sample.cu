#include "./assign_sample-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(AssignSampleParam param) {
      return new AssignSampleOp<gpu>(param);
    }
  }
}
