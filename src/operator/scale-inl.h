/*!
 * Copyright (c) 2015 by Contributors
 * \file scale-inl.h
 * \brief scale operator and symbol
*/
#ifndef MXNET_OPERATOR_SCALE_INL_H_
#define MXNET_OPERATOR_SCALE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace scale {
enum ScaleOpInputs {kData, kGamma, kBeta};
enum ScaleOpOutputs {kOut};
}  // namespace scale

struct ScaleParam : public dmlc::Parameter<ScaleParam> {
  bool no_bias;
  DMLC_DECLARE_PARAMETER(ScaleParam) {
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
  }
};

template<typename xpu>
class ScaleOp : public Operator {
 public:
  explicit ScaleOp(ScaleParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[scale::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[scale::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[scale::kData].shape_[0],
                               in_data[scale::kData].shape_[1], 1, 1);
      data = in_data[scale::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[scale::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[scale::kData].get<xpu, 4, real_t>(s);
      out = out_data[scale::kOut].get<xpu, 4, real_t>(s);
    }
    Tensor<xpu, 1> slope = in_data[scale::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[scale::kBeta].get<xpu, 1, real_t>(s);

    Assign(out, req[scale::kOut], broadcast<1>(slope, data.shape_) * data +
           broadcast<1>(bias, data.shape_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_grad.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;

    if (in_data[scale::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[scale::kOut].shape_[0],
                               out_grad[scale::kOut].shape_[1], 1, 1);
      data = in_data[scale::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[scale::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[scale::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[scale::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[scale::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[scale::kData].get<xpu, 4, real_t>(s); 
    }

    Tensor<xpu, 1> slope = in_data[scale::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[scale::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[scale::kBeta].get<xpu, 1, real_t>(s);

    Assign(gslope, req[scale::kGamma], sumall_except_dim<1>(grad * data));
    Assign(gbias, req[scale::kBeta], sumall_except_dim<1>(grad));
    Assign(grad_in, req[scale::kData], (grad * broadcast<1>(slope, data.shape_)));
  }

 private:
  ScaleParam param_;

};  // class ScaleOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator *CreateOp(ScaleParam param);

#if DMLC_USE_CXX11
class ScaleProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ScaleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Scale";
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ScaleParam param_;
};  // class ScaleParam

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SCALE_INL_H_
