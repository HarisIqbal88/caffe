#ifndef CAFFE_MYDIFFUSION2_LAYER_HPP_
#define CAFFE_MYDIFFUSION2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {


template <typename Dtype>
class MyDiffusion2Layer : public Layer<Dtype> {
 public:
  explicit MyDiffusion2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual inline const char* type() const { return "MyDiffusion2"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void Weights_Reshape_Forward(const Dtype* bottom_1, Dtype* weights, 
        int im_height, int im_width, const int* seqX, const int* seqY);
  void Weights_Reshape_Backward(const Dtype* bottom_1, Dtype* weights, 
    int im_height, int im_width, const int* seqX, const int* seqY);
  void Data_Reshape_Forward(const Dtype* bottom, Dtype* data_in, 
        int im_height, int im_width, const int* seqX, const int* seqY);
  void Data_Reshape_Backward(const Dtype* bottom, Dtype* data_in, 
        int im_height, int im_width, const int* seqX, const int* seqY);
      
  vector<int> strArr2intVec(string str);
  
  int num_iter_;
  int im_height;
  int im_width;
  int channels_;
  int ne_;
  int num_;
  int normalize_skip;
  int normalize_skip_start;

 private:
  Blob<Dtype> data_in_;
  Blob<Dtype> weights_;
  Blob<Dtype> col_buffer_;

  Blob<Dtype> data_in_b;
  Blob<Dtype> weights_b;
  Blob<Dtype> col_buffer_b;

  Blob<Dtype> normalization_;
  Blob<Dtype> normalization2_;
  Blob<Dtype> alpha_;  
  Blob<Dtype> ONEvector_ne;
  Blob<Dtype> ONEvector_ch;
   
  Blob<int> seqX;
  Blob<int> seqY;
  
};

}  // namespace caffe

#endif  // CAFFE_MYDIFFUSION2_LAYER_HPP_
