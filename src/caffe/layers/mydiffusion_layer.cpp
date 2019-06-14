#include <vector>

#include "caffe/layers/mydiffusion_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Helping functions
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
vector<int> MyDiffusionLayer<Dtype>::strArr2intVec(string str)
{
    vector<int> arr;
    string tmp;
    for (int i=0;i<str.length();i++){
        if (str[i]==',') {
            arr.push_back( atoi(tmp.c_str()) );
            tmp = "";}
        else if (i==(str.length()-1)) {
            tmp.push_back(str[i]);
            arr.push_back(atoi(tmp.c_str()));}
        else{
            tmp.push_back(str[i]);}
    }
    return arr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Layer SetUp
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  MyDiffusionParameter param = this->layer_param_.mydiffusion_param();
  
  LOG(INFO)<<"PROPAGATE_DOWN[0]:   "<<this->param_propagate_down(0);
  LOG(INFO)<<"PROPAGATE_DOWN[1]:   "<<this->param_propagate_down(1);
  LOG(INFO)<<"PROPAGATE_DOWN[2]:   "<<this->param_propagate_down(2);

  num_iter_    = param.iter(); 
  tanh_nonlinearity = param.istanh();
  a = param.tanh_scale();
  tanh_skip = param.tanh_skip();
  tanh_skip_start = param.tanh_skip_start();
  
  vector<int> tmpX_, tmpY_;
  tmpX_ = strArr2intVec( param.seq_x() );
  tmpY_ = strArr2intVec( param.seq_y() );
  ne_  = tmpX_.size();
  
  channels_ = bottom[0]->count(1,2);
  
  ONEvector_ne.Reshape(1,1,1,ne_);
  caffe_set(ONEvector_ne.count(), Dtype(1), ONEvector_ne.mutable_cpu_data());
  
  seqX.Reshape(1,1,1,ne_);
  seqY.Reshape(1,1,1,ne_);  
  int* x = seqX.mutable_cpu_data();
  int* y = seqY.mutable_cpu_data();  
  for(int i=0;i<ne_;i++){
    x[i] = tmpX_[i];
    y[i] = tmpY_[i];
  }
  
  CHECK_EQ(ne_, bottom[1]->count(1,2) ) << "num of neighbours should be equal to 2nd dim of weights blob"; 

  
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Reshape
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

 // LOG(INFO) << "param" << this->layer_param_.blobs_size() << "     got it?";
  im_height = bottom[0]->height();
  im_width  = bottom[0]->width();
  num_      = bottom[0]->count(2); //im_width*im_height;

  CHECK_GT(num_iter_, 0) << "number of iterations needs to be positive.";

  data_in_.Reshape(1,channels_,num_,ne_);
  weights_.Reshape(1,1,num_,ne_);
  col_buffer_.Reshape(1,num_iter_,channels_,num_);

  data_in_b.Reshape(1,channels_,num_,ne_);
  weights_b.Reshape(1,1,num_,ne_);
  col_buffer_b.Reshape(1,num_iter_,channels_,num_);

  normalization_.Reshape(1,1,1,num_);

  top[0]->ReshapeLike(*bottom[0]);  
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Data Reshape Forward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::Data_Reshape_Forward(const Dtype* bottom, Dtype* data_in, 
        int im_height, int im_width, const int* seqX, const int* seqY) {
  int ind = 0;
  int pos = 0;
  for (int i = 0; i < (im_height); ++i) 
  {
    ind = i * im_width;
    for (int j = 0; j < (im_width); ++j) 
    {
      pos = ind + j;
      for (int k=0;k<ne_;k++)
      {
        if (!( (j+seqX[k])<0 || (j+seqX[k])>=im_width || (i+seqY[k])<0 || (i+seqY[k])>=im_height ))
        {
            data_in[ne_*pos+k]=bottom[ pos + seqY[k]*im_width + seqX[k] ]; 
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Data Reshape Backward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::Data_Reshape_Backward(const Dtype* bottom, Dtype* data_in, 
        int im_height, int im_width, const int*  seqX, const int* seqY) {
  int ind = 0;
  int pos = 0;
  for (int i = 0; i < (im_height); ++i) 
  {
    ind = i * im_width;
    for (int j = 0; j < (im_width); ++j) 
    {
      pos = ind + j;
      for (int k=0;k<ne_;k++)
      {
        if (!( (j-seqX[k])<0 || (j-seqX[k])>=im_width || (i-seqY[k])<0 || (i-seqY[k])>=im_height ))
        {
            data_in[ne_*pos+k]=bottom[ pos - (seqY[k]*im_width + seqX[k]) ]; 
        }
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Weights Reshape Forward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::Weights_Reshape_Forward(const Dtype* bottom_1, Dtype* weights, 
        int im_height, int im_width, const int* seqX, const int*  seqY) {
  int ind = 0;
  int pos = 0;
  for (int i = 0; i < (im_height); ++i) 
  {
    ind = i * im_width;
    for (int j = 0; j < (im_width); ++j) 
    {
      pos = ind + j;
      for (int k=0;k<ne_;k++)
      {
        if (!( (j+seqX[k])<0 || (j+seqX[k])>=im_width || (i+seqY[k])<0 || (i+seqY[k])>=im_height ))
        { 
            weights[ne_*pos+k]=bottom_1[ pos + k*num_ ];
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Weights Reshape Backwards
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void MyDiffusionLayer<Dtype>::Weights_Reshape_Backward(const Dtype* bottom_1, Dtype* weights, 
    int im_height, int im_width, const int* seqX, const int* seqY) {
  int ind = 0;
  int pos = 0;
  for (int i = 0; i < (im_height); ++i) 
  {
    ind = i * im_width;
    for (int j = 0; j < (im_width); ++j) 
    {
      pos = ind + j;
      for (int k=0;k<ne_;k++)
      {
        if (!( (j-seqX[k])<0 || (j-seqX[k])>=im_width || (i-seqY[k])<0 || (i-seqY[k])>=im_height ))
        { 
            weights[ne_*pos+k]=bottom_1[ pos - ( seqY[k]*im_width + seqX[k] ) +k*num_];
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Forward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void MyDiffusionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Dtype* top_data  = top[0]->mutable_cpu_data();
  const Dtype* bottom_0  = bottom[0]->cpu_data();
  const Dtype* bottom_1  = bottom[1]->cpu_data();
  
  Dtype* data_in   = data_in_.mutable_cpu_data();
  Dtype* weights   = weights_.mutable_cpu_data();
  Dtype* col_buffer= col_buffer_.mutable_cpu_data();

  Weights_Reshape_Forward(bottom_1 , weights , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
  Dtype b = 1/tanh(a);
  for (int itr=0;itr<num_iter_;itr++){
  for (int ch=0;ch<channels_;ch++){    
    if (itr==0)    
      Data_Reshape_Forward(bottom_0 + ch*num_, data_in , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
    else
      Data_Reshape_Forward(top_data + ch*num_, data_in , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
    for(int i=0;i<num_;i++)
      *(col_buffer + itr*num_*channels_ + ch*num_+i) = caffe_cpu_dot(ne_, data_in+ne_*i, weights+ne_*i);
  }
  for(int i=0;i<(channels_*num_);i++){
    if (tanh_nonlinearity && !( (itr-tanh_skip_start+1)%tanh_skip )){
      *(top_data+i) = b*tanh(a*col_buffer[itr*num_*channels_ + i]);}
    else
      *(top_data+i) = col_buffer[itr*num_*channels_ + i];          
  }
  }  
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Backward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_1  = bottom[1]->cpu_data();
  
  Dtype* data_in   = data_in_.mutable_cpu_data();
  Dtype* weights   = weights_.mutable_cpu_data();
  Dtype* col_buffer= col_buffer_.mutable_cpu_data();

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  Weights_Reshape_Backward(bottom_1 , weights , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
 
  Dtype c = 2*a;
  Dtype d = c / tanh(a);
  
  if(propagate_down[0]){ 
  for(int itr=(num_iter_-1);itr>=0;itr--){
    if(tanh_nonlinearity && !( (itr-tanh_skip_start+1)%tanh_skip )){  //Add condition to non-linearlize after some iterations
        caffe_scal(num_*channels_, c, col_buffer + itr*num_*channels_);  // y-> 2*a*y
        for(int j=0;j<(num_*channels_);j++)
            col_buffer[j + itr*num_*channels_] = cosh(col_buffer[j + itr*num_*channels_])+1;         // y-> 2*a*y -> cosh(2*a*y)+1
        if(itr==(num_iter_-1))
            caffe_div(num_*channels_, top_diff, col_buffer + itr*channels_*num_,col_buffer + itr*channels_*num_);  // y-> 2*a*y -> cosh(2*a*y)+1 -> E_z./{cosh(2*a*y)+1}
        else
            caffe_div(num_*channels_, bottom_diff, col_buffer + itr*channels_*num_,col_buffer + itr*channels_*num_); // y-> 2*a*y -> cosh(2*a*y)+1 -> E_z./{cosh(2*a*y)+1}
        caffe_scal(num_*channels_, d, col_buffer + itr*channels_*num_);  // y-> 2*a*y -> cosh(2*a*y)+1 -> E_z./{cosh(2*a*y)+1} -> d*E_z./{cosh(2*a*y)+1}
    }
    else{
        if(itr==(num_iter_-1))
            caffe_copy(num_*channels_, top_diff, col_buffer + itr*num_*channels_);
        else
            caffe_copy(num_*channels_, bottom_diff, col_buffer + itr*num_*channels_);
       }
    for (int ch=0;ch<channels_;ch++){
        Data_Reshape_Backward(col_buffer + itr*channels_*num_ + ch*num_, data_in , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());     
        for(int i=0;i<num_;i++)
            *(bottom_diff + ch*num_ + i) = caffe_cpu_dot(ne_, data_in+ne_*i, weights+ne_*i);
    }
  }
  }
  //check how to block propagate_down for weights 

}
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef CPU_ONLY
STUB_GPU(MyDiffusionLayer);
#endif

INSTANTIATE_CLASS(MyDiffusionLayer);
REGISTER_LAYER_CLASS(MyDiffusion);

}  // namespace caffe
