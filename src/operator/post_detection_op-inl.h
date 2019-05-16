// post_detection_op
// PostDetectionOp

#ifndef MXNET_OPERATOR_PostDetectionOp_INL_H_
#define MXNET_OPERATOR_PostDetectionOp_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace post_detection_op_enum {
enum PostDetectionOpInputs {kRois, kScores, kBbox_deltas};
enum PostDetectionOpOutputs {kBatch_boxes, kBatch_boxes_rois};
enum PostDetectionOpResource {kTempSpace};
}

struct PostDetectionOpParam : public dmlc::Parameter<PostDetectionOpParam> {
  float im_w, im_h, im_c;
  float thresh;
  int n_classes;
  int input_bbox_num;
  int max_output_num;
  float nms_thresh_lo, nms_thresh_hi;
  DMLC_DECLARE_PARAMETER(PostDetectionOpParam) {
    DMLC_DECLARE_FIELD(im_w)
      .set_default(640.0).describe("Image width.");
	 DMLC_DECLARE_FIELD(im_h)
      .set_default(360.0).describe("Image height.");
    DMLC_DECLARE_FIELD(im_c)
      .set_default(3.0).describe("Image channel.");
  	DMLC_DECLARE_FIELD(thresh)
      .set_default(0.9).describe("Threshold.");
  	DMLC_DECLARE_FIELD(n_classes)
      .set_default(7.0).describe("Number of classes.");
  	DMLC_DECLARE_FIELD(nms_thresh_lo)
      .set_default(0.3).describe("Lower bound of NMS.");
    DMLC_DECLARE_FIELD(nms_thresh_hi)
      .set_default(0.5).describe("Higher bound of NMS.");
    DMLC_DECLARE_FIELD(input_bbox_num)
      .set_default(100).describe("Number of raw input bounding boxes");
    DMLC_DECLARE_FIELD(max_output_num)
      .set_default(100).describe("Max number of output after nms of each image");
  }
};

template<typename xpu, typename DType>
class PostDetectionOp : public Operator {
 public:
  explicit PostDetectionOp(PostDetectionOpParam param) {
    this->param_ = param;    
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_EQ(req.size(), 2);
    CHECK_EQ(req[post_detection_op_enum::kBatch_boxes], kWriteTo);
    CHECK_EQ(req[post_detection_op_enum::kBatch_boxes_rois], kWriteTo);

    int batch_size = in_data[post_detection_op_enum::kScores].size(0);
    int n_classes = this->param_.n_classes;
    int input_bbox_num = this->param_.input_bbox_num;
    int max_output_num = this->param_.max_output_num;
    // rois = in_data[0].asnumpy()[:, 1:]
    // scores = in_data[1].asnumpy()
    // bbox_deltas = in_data[2].asnumpy()
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Shape<2> roi_shape = Shape2(batch_size*input_bbox_num, 5);
    Tensor<xpu, 2, DType> rois = 
      in_data[post_detection_op_enum::kRois].get_with_shape<xpu, 2, DType>(roi_shape, s);
    Tensor<xpu, 3, DType> scores = in_data[post_detection_op_enum::kScores]
                                    .get_with_shape<xpu, 3, DType>(Shape3(batch_size, input_bbox_num, n_classes), s);
    Tensor<xpu, 3, DType> bbox_deltas = in_data[post_detection_op_enum::kBbox_deltas]
                                    .get_with_shape<xpu, 3, DType>(Shape3(batch_size, input_bbox_num, 4*n_classes), s);
    // self.batch_boxes = np.zeros(shape=(self._im_shape[0], len(self._classes), 20, 5))
    // self.batch_boxes_rois = np.zeros(shape=(100, 5))
    Tensor<xpu, 3, DType> batch_boxes = out_data[post_detection_op_enum::kBatch_boxes]
                                    .get_with_shape<xpu, 3, DType>(Shape3(batch_size, max_output_num, 6), s);
    Tensor<xpu, 2, DType> batch_boxes_rois = out_data[post_detection_op_enum::kBatch_boxes_rois]
                                    .get_with_shape<xpu, 2, DType>(Shape2(batch_size * max_output_num, 5), s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    PostDetctionForward(
      rois, scores, bbox_deltas, // inputs
      batch_boxes, batch_boxes_rois, // outputs
      this->param_
    );
    if (ctx.is_train) {
    	LOG(FATAL) << "Should use in test mode only.";
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
	  LOG(FATAL) << "PostDetectionOp Backward Not Implemented.";
  }

 private:
  PostDetectionOpParam param_;
};  // class LSoftmaxOp

template<typename xpu>
Operator *CreateOp(PostDetectionOpParam param, int dtype);

#if DMLC_USE_CXX11
	class PostDetectionOpProp : public OperatorProperty {
	public:
	  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
      // std::cout << "Init ... ";
	    param_.Init(kwargs);
      // std::cout << " Done" << std::endl;
	  }

    std::map<std::string, std::string> GetParams() const override {
      // std::cout << "GetParams" << std::endl;
 	   	return param_.__DICT__();
  	}

	  std::vector<std::string> ListArguments() const override {
    	return {"rois", "scores", "bbox_deltas"};
  	}

	  std::vector<std::string> ListOutputs() const override {
    	return {"batch_boxes", "batch_boxes_rois"};
  	}

	  int NumOutputs() const override {
	    return 2;
	  }

	  int NumVisibleOutputs() const override {
	    return 2;
	  }

    bool InferShape(std::vector<TShape> *in_shape,
	                  std::vector<TShape> *out_shape,
	                  std::vector<TShape> *aux_shape) const override {
	    using namespace mshadow;
	    CHECK_EQ(in_shape->size(), 3) << "Input:[rois, scores, bbox_deltas]";
	    const TShape &roi_shape = in_shape->at(post_detection_op_enum::kRois);
	    const TShape &score_shape = in_shape->at(post_detection_op_enum::kScores);
			const TShape &bbox_deltas_shape = in_shape->at(post_detection_op_enum::kBbox_deltas);

	    CHECK_EQ(roi_shape.ndim(), 2) << "roi_shape should be (batch_size*100, 5)";
	    CHECK_EQ(score_shape.ndim(), 3) << "score_shape should be (batch_size, 100, 7)";
	    CHECK_EQ(bbox_deltas_shape.ndim(), 3) << "score_shape should be (batch_size, 100, 28)";

	    const int batch_size = score_shape[0];
	    out_shape->clear();
	    out_shape->push_back(Shape3(batch_size, param_.max_output_num, 6));  // batch_boxes
      out_shape->push_back(Shape2(batch_size * param_.max_output_num, 5));  // batch_boxes_rois
	    aux_shape->clear();
	    return true;
	  }

 	  std::string TypeString() const override {
		  return "PostDetection";
		}

	  OperatorProperty *Copy() const override {
	    auto ptr = new PostDetectionOpProp();
	    ptr->param_ = param_;
	    return ptr;
	  }

    Operator *CreateOperator(Context ctx) const override {
	    LOG(FATAL) << "Not Implemented.";
	    return NULL;
	  }

    Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
  		                         std::vector<int> *in_type) const override;

	private:
  	PostDetectionOpParam param_;
	};	
#endif // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_PostDetectionOp_INL_H_