/*!
 * Copyright (c) 2015 by Contributors
 * \file assign_sample-inl.h
 * \brief assign negative samples
 * \author Xuan Lin
*/
#ifndef MXNET_OPERATOR_ASSIGN_SAMPLE_INL_H_
#define MXNET_OPERATOR_ASSIGN_SAMPLE_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {
namespace assignsample {
  enum AssignSampleOpInputs {kClsdata, kRegdata, kLabel};
  enum AssignSampleOpOutputs {kClsout, kRegout, kMaskout};
  enum AssignSampleOpResource {kTempSpace};
}

struct AssignSampleParam : public dmlc::Parameter<AssignSampleParam> {
  bool use_negative_mining;
  int anchor_num;
  int fea_shape;
  int batch_size;
  DMLC_DECLARE_PARAMETER(AssignSampleParam) {
  DMLC_DECLARE_FIELD(use_negative_mining).set_default(false)
  .describe("use negative mining in training");
  DMLC_DECLARE_FIELD(anchor_num).describe("anchor number in every grid of feature map");
  DMLC_DECLARE_FIELD(fea_shape).describe("first dim of feature map");
  DMLC_DECLARE_FIELD(batch_size).describe("num of sampled anchors every train image");
  }
};

typedef std::pair<int, float> m_pair;
static bool Comparator(const m_pair& l, const m_pair& r) {
  return l.second > r.second;
}


template <typename xpu>
class AssignSampleOp : public Operator {
 public:
  explicit AssignSampleOp(AssignSampleParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[assignsample::kClsdata].size(0);
    int k = in_data[assignsample::kClsdata].size(1);
    int m = static_cast<int>(in_data[assignsample::kClsdata].Size()/n/k);
    Shape<2> s2 = Shape2(n, m);
    Shape<3> s3 = Shape3(n, k, m);
    Shape<4> s4 = Shape4(n, param_.anchor_num * 4, param_.fea_shape,
        (in_data[assignsample::kRegdata].Size()/param_.anchor_num/param_.fea_shape/4));
    Tensor<xpu, 3> clsdata = in_data[assignsample::kClsdata]
        .get_with_shape<xpu, 3, real_t>(s3, s);
    Tensor<xpu, 4> regdata = in_data[assignsample::kRegdata]
        .get_with_shape<xpu, 4, real_t>(s4, s);
    Tensor<xpu, 2> label = in_data[assignsample::kLabel]
        .get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 3> clsout = out_data[assignsample::kClsout]
        .get_with_shape<xpu, 3, real_t>(s3, s);
    Tensor<xpu, 4> regout = out_data[assignsample::kRegout]
        .get_with_shape<xpu, 4, real_t>(s4, s);
    Tensor<xpu, 2> maskout = out_data[assignsample::kMaskout]
        .get_with_shape<xpu, 2, real_t>(s2, s);
    maskout = 0.0f;

    Tensor<cpu, 2> maskout_cpu = NewTensor<cpu, real_t>(s2, 0.0f);

    Tensor<xpu, 1> score = clsdata[0][1];
    Tensor<cpu, 1> score_cpu = NewTensor<cpu, real_t>(score.shape_, 0.0f);
    Copy(score_cpu, score, s);

    Tensor<cpu, 2> label_cpu = NewTensor<cpu, real_t>(s2, 0.0f);
    Copy(label_cpu, label, s);

    std::vector<int> bg_idx;
    std::vector<m_pair> i_s_pair;
    int fg_num = 0;

    for (int i = 0; i < label.size(1); ++i) {
      if (label_cpu[0][i] == 1) {
        fg_num += 1;
        maskout_cpu[0][i] = 1.0f;
      } else {
        if (label_cpu[0][i] == 0) {
          bg_idx.push_back(i);
          i_s_pair.push_back(m_pair(i, score_cpu[i]));
        }
      }
    }
    int bg_num = param_.batch_size - fg_num;
    // negative mining
    if (param_.use_negative_mining) {
      std::sort(i_s_pair.begin(), i_s_pair.end(), Comparator);
      for (int i = 0; i < bg_num; ++i) {
        maskout_cpu[0][i_s_pair[i].first] = 1.0f;
      }
    } else {
      std::random_shuffle(bg_idx.begin(), bg_idx.end());
      for (int i = 0; i < bg_num; ++i) {
        maskout_cpu[0][bg_idx[i]] = 1.0f;
      }
    }
    Copy(maskout, maskout_cpu, s);

    Shape<4> dshape = Shape4(1, param_.anchor_num, param_.fea_shape,
        out_data[assignsample::kMaskout].Size()/param_.anchor_num/param_.fea_shape);
    Tensor<cpu, 4> mask_reshape = NewTensor<cpu, real_t>(dshape, 0.0f);
    mask_reshape = reshape(maskout_cpu, dshape);
    Tensor<xpu, 4> regmask = ctx.requested[assignsample::kTempSpace].get_space<xpu>(s4, s);
    Tensor<cpu, 4> regmask_cpu = NewTensor<cpu, real_t>(s4, 0.0f);
    for (int i = 0; i < param_.anchor_num; ++i) {
      for (int j = i*4; j < (i+1)*4; ++j) {
        regmask_cpu[0][j] += mask_reshape[0][i];
      }
    }
    Copy(regmask, regmask_cpu, s);
    Assign(clsout, req[assignsample::kClsout], clsdata * 1);
    Assign(regout, req[assignsample::kRegout], regdata * regmask);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = out_data[assignsample::kClsout].size(0);
    int k = out_data[assignsample::kClsout].size(1);
    int m = static_cast<int>(out_data[assignsample::kClsout].Size()/n/k);

    Shape<2> s2 = Shape2(n, m);
    Shape<3> s3 = Shape3(n, k, m);
    Shape<4> s4 = Shape4(n, param_.anchor_num * 4, param_.fea_shape,
        out_data[assignsample::kRegout].Size()/param_.anchor_num/param_.fea_shape/4);
    Tensor<xpu, 3> clsin_grad = in_grad[assignsample::kClsdata]
        .get_with_shape<xpu, 3, real_t>(s3, s);
    Tensor<xpu, 4> regin_grad = in_grad[assignsample::kRegdata]
        .get_with_shape<xpu, 4, real_t>(s4, s);
    Tensor<xpu, 2> maskout = out_data[assignsample::kMaskout]
        .get_with_shape<xpu, 2, real_t>(s2, s);

    Tensor<cpu, 2> maskout_cpu = NewTensor<cpu, real_t>(s2, 0.0f);
    Copy(maskout_cpu, maskout, s);
    int p_count = 0;
    for (int i = 0; i < maskout_cpu.size(1); ++i) {
      if (maskout_cpu[0][i] == 1) { p_count += 1; }
    }

    Shape<2> dshape2 = Shape2(2, out_data[assignsample::kMaskout].Size());
    Tensor<xpu, 2> mask_rep = ctx.requested[assignsample::kTempSpace].get_space<xpu>(dshape2, s);
    mask_rep = 0.0f;
    Tensor<xpu, 1> mask = maskout[0];
    mask_rep = repmat(mask, 2);
    Assign(clsin_grad, req[assignsample::kClsdata], reshape(mask_rep, s3)/p_count);

    Shape<4> dshape4 = Shape4(1, param_.anchor_num, param_.fea_shape,
        out_data[assignsample::kMaskout].Size()/param_.anchor_num/param_.fea_shape);
    Tensor<xpu, 4> mask_reshape = ctx.requested[assignsample::kTempSpace]
        .get_space<xpu>(dshape4, s);
    Tensor<cpu, 4> mask_reshape_cpu = NewTensor<cpu, real_t>(dshape4, 0.0f);
    mask_reshape = reshape(maskout, dshape4);
    Copy(mask_reshape_cpu, mask_reshape, s);
    Tensor<xpu, 4> regmask = ctx.requested[assignsample::kTempSpace].get_space<xpu>(s4, s);
    Tensor<cpu, 4> regmask_cpu = NewTensor<cpu, real_t>(s4, 0.0f);
    for (int i = 0; i < param_.anchor_num; ++i) {
      for (int j = i*4; j < (i+1)*4; ++j) {
        regmask_cpu[0][j] += mask_reshape_cpu[0][i];
      }
    }
    Copy(regmask, regmask_cpu, s);
    Assign(regin_grad, req[assignsample::kRegdata], regmask * 1);
  }

 private:
  AssignSampleParam param_;
};

template<typename xpu>
Operator* CreateOp(AssignSampleParam param);

#if DMLC_USE_CXX11
class AssignSampleProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"clsdata", "regdata", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"clsout", "regout", "maskout"};
  }

  int NumOutputs() const override {
    return 3;
  }

  void Init(const std::vector<std::pair<std::string, std::string>>& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[clsdata, regdata, label]";
    const TShape &clsshape = in_shape->at(assignsample::kClsdata);
    const TShape &regshape = in_shape->at(assignsample::kRegdata);
    const TShape &labelshape = in_shape->at(assignsample::kLabel);
    if (clsshape.ndim() == 0) return false;
    if (regshape.ndim() == 0) return false;
    const TShape &lshape = Shape2(clsshape[0], clsshape[2]);
    if (labelshape[0] != lshape[0] || labelshape[1] != lshape[1]) {
      std::ostringstream os;
      os << "Shape inconsistent, Provided " << '='<< labelshape << ','
         << "inferred shape =" << lshape;
      throw ::mxnet::op::InferShapeError(os.str(), 1);
    }
    out_shape->clear();
    out_shape->push_back(clsshape);
    out_shape->push_back(regshape);
    out_shape->push_back(labelshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AssignSampleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "AssignSample";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_data[assignsample::kClsout],
            out_data[assignsample::kRegout], out_data[assignsample::kMaskout]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[assignsample::kClsout], in_grad[assignsample::kClsdata]},
           {out_data[assignsample::kRegout], in_grad[assignsample::kRegdata]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[assignsample::kClsdata], out_data[assignsample::kClsout]},
            {in_data[assignsample::kRegdata], out_data[assignsample::kRegout]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }
  Operator* CreateOperator(Context ctx) const override;

 protected:
  AssignSampleParam param_;
};  // class AssignSampleProperty
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ASSIGN_SAMPLE_INL_H_
