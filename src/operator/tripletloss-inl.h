/*
 * lbh-inl-new.h
 *
 *  Created on: Nov 28, 2018
 *      Author: tingyu.mao
 */

#ifndef MXNET_OPERATOR_TRIPLET_H_
#define MXNET_OPERATOR_TRIPLET_H_

#define MAX(a, b) ((a>b) ? a:b)
#define MIN(a, b) ((a<b) ? a:b)

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace tripletloss_enum {
enum TripletOpInputs {kData, kLabel};
enum TripletOpOutputs {kOut};
enum TripletOpResource {kTempSpace};
}

// hyper-parameters
struct TripletLossParam : public dmlc::Parameter<TripletLossParam> {

	float margin;
	float epsilon;
	bool out_grad;

	DMLC_DECLARE_PARAMETER(TripletLossParam) {

		DMLC_DECLARE_FIELD(margin).set_default(1.0)
		.describe("Margin between pos/neg pair");

		DMLC_DECLARE_FIELD(epsilon).set_default(1.0)
		.describe("The weighted term of triple loss");

		DMLC_DECLARE_FIELD(out_grad).set_default(false)
		.describe("Multiplies gradient with output gradient element-wise.");

	}
};

// operator
template<typename xpu, typename DType>
class TripletLossOp : public Operator {
public:
	// explicit constructor to avoid implicit type conversion
	explicit TripletLossOp(TripletLossParam param) : param_(param) {}

	// forward function
	virtual void Forward(const OpContext &ctx,
			             const std::vector<TBlob> &in_data,
			             const std::vector<OpReqType> &req,
			             const std::vector<TBlob> &out_data,
			             const std::vector<TBlob> &aux_args) {
		using namespace mshadow;
		using namespace mshadow::expr;

		CHECK_EQ(in_data.size(), 2) << "Triplet Loss Input: [data, label]";
		CHECK_EQ(out_data.size(), 1) << "Triplet Loss Output: [loss]";
		CHECK_EQ(req.size(), 1) << "Req size should be consistent with the output size";

		CHECK(req[tripletloss_enum::kOut] == kNullOp || req[tripletloss_enum::kOut] == kWriteTo);

		Stream<xpu> *s = ctx.get_stream<xpu>();

        const int n = in_data[tripletloss_enum::kData].size(0); // batch size
        const int k = in_data[tripletloss_enum::kData].size(1); // feature number

        Tensor<xpu, 2, DType> data = in_data[tripletloss_enum::kData].get_with_shape<xpu, 2, DType>(Shape2(n, k), s);
        Tensor<xpu, 1, DType> label = in_data[tripletloss_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
        Tensor<xpu, 1, DType> out = out_data[tripletloss_enum::kOut].get_with_shape<xpu, 1, DType>(Shape1(n), s);

        // TODO: consider inplace option to promote speed
        Tensor<xpu, 2, DType> dst = ctx.requested[tripletloss_enum::kTempSpace].get_space_typed<xpu, 2, DType>(Shape2(n, n), s);

        // L2 distance
        // TODO: any other better solution in mshadow?
        dst[0] = sumall_except_dim<0>(data * data);
        dst = repmat(dst[0], n);
        dst += dst.T();
        dst *= 0.5;
        dst -= dot(data, data.T());

        TripletLossForward(dst, label, out, n, param_.margin, param_.epsilon);

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

		CHECK_EQ(out_grad.size(), 1);
		CHECK_EQ(in_data.size(), 2);
		CHECK_EQ(out_data.size(), 1);
		CHECK_GE(in_grad.size(), 1); // grad wrt data
		CHECK_GE(req.size(), 1); // should req be consistent with in_grad?
        CHECK_EQ(req[tripletloss_enum::kData], kWriteTo);

		Stream<xpu> *s = ctx.get_stream<xpu>();
		const int n = in_data[tripletloss_enum::kData].size(0); // batch size
		const int k = in_data[tripletloss_enum::kData].size(1); // feature number

		Tensor<xpu, 2, DType> data = in_data[tripletloss_enum::kData].get_with_shape<xpu, 2, DType>(Shape2(n, k), s);
		Tensor<xpu, 1, DType> label = in_data[tripletloss_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
		Tensor<xpu, 2, DType> data_grad = in_grad[tripletloss_enum::kData].get_with_shape<xpu, 2, DType>(Shape2(n, k), s);

		Tensor<xpu, 3, DType> workspace = ctx.requested[tripletloss_enum::kTempSpace].get_space_typed<xpu, 3, DType>(Shape3(2, n, n), s);
		TripletLossBackward(workspace, data, label, data_grad, n, k, param_.margin, param_.epsilon);

	}

private:
	TripletLossParam param_;

}; // class TripletLossOP

// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(TripletLossParam param, int dtype);

#if DMLC_USE_CXX11
class TripletLossProp : public OperatorProperty {
public:

	// init parameters
	void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
		param_.Init(kwargs);
	}

	// get params
	std::map<std::string, std::string> GetParams() const override {
		return param_.__DICT__();
	}

	// list arguments
	std::vector<std::string> ListArguments() const override {
		return {"data", "label"};
	}

	// list outputs
	std::vector<std::string> ListOutputs() const override {
	    return {"output"};
	}

	// number of outputs
	int NumOutputs() const override {
		return 1;
	}

	int NumVisibleOutputs() const override {
		return 1;
	}

	// infer output shape based on input shape
	bool InferShape(std::vector<TShape> *in_shape,
			        std::vector<TShape> *out_shape,
			        std::vector<TShape> *aux_shape) const override{
		using namespace mshadow;

		CHECK_EQ(in_shape->size(), 2) << "Input: [data, label]";
		const TShape &dshape = in_shape->at(tripletloss_enum::kData);
		const TShape &lshape = in_shape->at(tripletloss_enum::kLabel);

		CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, feature_dim)";
		CHECK_EQ(lshape.ndim(), 1) << "label shape should be (batch_size, )";

		const int n = dshape[0]; // batch_size
		const int feature_dim = dshape[1];

		out_shape->clear();
		out_shape->push_back(Shape1(n));

		return true;
	}

	std::vector<ResourceRequest> ForwardResource(const std::vector<TShape> &in_shape) const override {
		return {ResourceRequest::kTempSpace};
	}

	std::vector<ResourceRequest> BackwardResource(const std::vector<TShape> &in_shape) const override {
	    return {ResourceRequest::kTempSpace};
	}

	std::vector<int> DeclareBackwardDependency(
	      const std::vector<int> &out_grad,
	      const std::vector<int> &in_data,
	      const std::vector<int> &out_data) const override {
	    return {out_data[tripletloss_enum::kOut], in_data[tripletloss_enum::kData], in_data[tripletloss_enum::kLabel]};
	}

	std::string TypeString() const override {
		return "TripletLoss";
	}

	OperatorProperty* Copy() const override {
		auto ptr = new TripletLossProp();
		ptr->param_ = param_;
		return ptr;
	}

	Operator* CreateOperator(Context ctx) const override {
	    LOG(FATAL) << "Not Implemented.";
	    return NULL;
	}

	Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
	                           std::vector<int> *in_type) const override;

protected:
	  TripletLossParam param_;
}; // class TripletLossProp

#endif // DMLC_USE_CXX11


} // end of namespace op
} // end of namesapce mxnet

#endif /* MXNET_OPERATOR_TRIPLET_H_ */
