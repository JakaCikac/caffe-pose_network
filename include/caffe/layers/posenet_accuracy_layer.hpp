#ifndef POSENET_ACCURACY_LAYER_HPP
#define POSENET_ACCURACY_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PosenetAccuracyLayer : public Layer<Dtype>
{
public:
  explicit PosenetAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PosenetAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }

  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 1; }

protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- PosenetAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  Blob<Dtype> default_weight_;

  int image_width_;
  int image_height_;
  float distance_start_;
  float distance_stop_;
  float distance_step_;
  int num_steps_;
};

}  // namespace caffe

#endif  // POSENET_ACCURACY_LAYER_HPP
