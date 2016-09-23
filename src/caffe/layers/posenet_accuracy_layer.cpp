#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/posenet_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PosenetAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  image_width_ = this->layer_param_.posenet_accuracy_param().image_width();
  image_height_ = this->layer_param_.posenet_accuracy_param().image_height();
  distance_start_ = this->layer_param_.posenet_accuracy_param().distance_start();
  distance_stop_ = this->layer_param_.posenet_accuracy_param().distance_stop();
  distance_step_ = this->layer_param_.posenet_accuracy_param().distance_step();
  num_steps_ = static_cast<int>((distance_stop_ - distance_start_) / distance_step_) + 1;
  LOG(INFO) << "image size = " << image_width_ << "x" << image_height_;
  LOG(INFO) << "distance start:step:stop = " << distance_start_ << ":" << distance_step_ << ":" << distance_stop_;
  LOG(INFO) << "num evaluated distances = " << num_steps_;
}

template <typename Dtype>
void PosenetAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_GE(image_width_, 1)
      << "image_width must be 1 or greater.";
  CHECK_GE(image_height_, 1)
      << "image_height must be 1 or greater.";
  CHECK_GE(distance_start_, 0)
      << "distance_start must be positive.";
  CHECK_GE(distance_stop_, 0)
      << "distance_stop must be positive.";
  CHECK(distance_step_ > 0.0)
      << "distance_step must greater than 0.";
  CHECK_GE(num_steps_, 1)
      << "num_steps_ must be 1 or greater.";
  CHECK_EQ(distance_start_ + distance_step_ * (num_steps_ - 1), distance_stop_)
      << "distance steps must be evenly spaced.";
  const int num_predictions = bottom[0]->count();
  const int num_labels = bottom[1]->count();
  CHECK_EQ(num_predictions, num_labels)
      << "Label and prediction must have the same dimension.";
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))
        << "Weight and label must have the same dimension.";
  }
  vector<int> top_accuracy_shape(2);
  top_accuracy_shape[0] = 1;
  top_accuracy_shape[1] = num_steps_;
  vector<int> top_euclidean_distance_error_shape(0);
  top[0]->Reshape(top_accuracy_shape);
  top[1]->Reshape(top_euclidean_distance_error_shape);
  // Initialize default weights to 1.
  default_weight_.ReshapeLike(*bottom[0]);
  caffe_set(default_weight_.count(), Dtype(1), default_weight_.mutable_cpu_data());
}

template <typename Dtype>
void PosenetAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const Dtype* prediction = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weight = bottom.size() > 2 ? bottom[2]->cpu_data() : default_weight_.cpu_data();
  const int num_points = bottom[0]->count() >> 1;

  vector<int> num_accurate_predictions(num_steps_);
  Dtype sum_dist = 0;
  int num_points_evaluated = 0;

  for (int i = 0; i < num_points; ++i) {
    const int ix = i << 1;
    const int iy = ix + 1;

    if (weight[ix] == 0 &&
        weight[iy] == 0) {
        continue;
    }

    const Dtype diff_x = (prediction[ix] - label[ix]) * image_width_ * weight[ix];
    const Dtype diff_y = (prediction[iy] - label[iy]) * image_height_ * weight[iy];
    const Dtype dist = sqrt(diff_x * diff_x + diff_y * diff_y);

    float eval_dist = distance_start_;

    for (int j = 0; j < num_steps_; ++j) {
      if (dist <= eval_dist) {
        ++num_accurate_predictions[j];
      }
      eval_dist += distance_step_;
    }

    sum_dist += dist;
    ++num_points_evaluated;
  }

  for (int i = 0; i < num_steps_; ++i) {
    top[0]->mutable_cpu_data()[i] = static_cast<Dtype>(num_accurate_predictions[i]) / num_points_evaluated;
  }
  top[1]->mutable_cpu_data()[0] = sum_dist / num_points_evaluated;

    // PosenetAccuracyLayer layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PosenetAccuracyLayer);
REGISTER_LAYER_CLASS(PosenetAccuracy);

}  // namespace caffe
