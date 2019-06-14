#include <vector>

#include "caffe/layers/mydiffusion2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Helping functions
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
vector<int> MyDiffusion2Layer<Dtype>::strArr2intVec(string str)
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
void MyDiffusion2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  MyDiffusion2Parameter param = this->layer_param_.mydiffusion2_param();
  
  num_iter_    = param.iter(); 
  
  vector<int> tmpX_, tmpY_;
  tmpX_ = strArr2intVec( param.seq_x() );
  tmpY_ = strArr2intVec( param.seq_y() );
  ne_  = tmpX_.size();

  ONEvector_ne.Reshape(1,1,1,ne_);
  caffe_set(ONEvector_ne.count(),(Dtype)1.,ONEvector_ne.mutable_cpu_data());  
  
  channels_ = bottom[0]->count(1,2);    
  ONEvector_ch.Reshape(1,1,1,channels_);
  caffe_set(ONEvector_ch.count(),(Dtype)1.,ONEvector_ch.mutable_cpu_data());
  
  seqX.Reshape(1,1,1,ne_);
  seqY.Reshape(1,1,1,ne_);  
  int* x = seqX.mutable_cpu_data();
  int* y = seqY.mutable_cpu_data();  
  for(int i=0;i<ne_;i++){
    x[i] = tmpX_[i];
    y[i] = tmpY_[i];
  }
  CHECK_GT(num_iter_, 0) << "number of iterations needs to be positive."; 
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Reshape
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusion2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  im_height = bottom[0]->height();
  im_width  = bottom[0]->width();
  num_      = bottom[0]->count(2); //im_width*im_height;  
  
  CHECK_GT(num_iter_, 0) << "number of iterations needs to be positive.";
  
  data_in_.Reshape(1,1,num_,ne_);
  weights_.Reshape(1,1,num_,ne_);
  col_buffer_.Reshape(1,num_iter_,channels_,num_);

  data_in_b.Reshape(1,1,num_,ne_);
  weights_b.Reshape(1,1,num_,ne_);
  col_buffer_b.Reshape(1,num_iter_,channels_,num_);

  normalization_.Reshape(1,1,1,num_);
  normalization2_.Reshape(1,1,1,num_);
  alpha_.Reshape(1,1,1,num_);
 
  top[0]->ReshapeLike(*bottom[0]);
  //LOG(INFO) << "Reshape is done" << bottom[0]->count(2) << "     got it?";   
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Data Reshape Forward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusion2Layer<Dtype>::Data_Reshape_Forward(const Dtype* bottom, Dtype* data_in, 
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
void MyDiffusion2Layer<Dtype>::Data_Reshape_Backward(const Dtype* bottom, Dtype* data_in, 
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
void MyDiffusion2Layer<Dtype>::Weights_Reshape_Forward(const Dtype* bottom_1, Dtype* weights, 
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
void MyDiffusion2Layer<Dtype>::Weights_Reshape_Backward(const Dtype* bottom_1, Dtype* weights, 
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
void MyDiffusion2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  Dtype* top_data  = top[0]->mutable_cpu_data();
  const Dtype* bottom_0  = bottom[0]->cpu_data();
  const Dtype* bottom_1  = bottom[1]->cpu_data();
  
  Dtype* data_in   = data_in_.mutable_cpu_data();
  Dtype* weights   = weights_.mutable_cpu_data();
  Dtype* col_buffer= col_buffer_.mutable_cpu_data();
  Dtype* Normalization = normalization_.mutable_cpu_data();
  
  Weights_Reshape_Forward(bottom_1 , weights , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
  
  for (int itr=0;itr<num_iter_;itr++){
  for (int ch=0;ch<channels_;ch++){    
    if (itr==0)    
      Data_Reshape_Forward(bottom_0 + ch*num_, data_in , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
    else
      Data_Reshape_Forward(top_data + ch*num_, data_in , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
    for(int i=0;i<num_;i++)
      *(col_buffer + itr*num_*channels_ + ch*num_ + i) = caffe_cpu_dot(ne_, data_in+ne_*i, weights+ne_*i);
  }
  if(!( (itr-normalize_skip_start+1)%normalize_skip ))
    for(int i=0;i<num_;i++)
        *(Normalization+i) = caffe_cpu_strided_dot(channels_, col_buffer + itr*num_*channels_ + i, num_, ONEvector_ch.cpu_data() , 1);
  for(int ch=0;ch<channels_;ch++){
    if(!( (itr-normalize_skip_start+1)%normalize_skip ))
        caffe_div(num_ , col_buffer + itr*num_*channels_ + ch*num_, Normalization, top_data + ch*num_);        
    else
        caffe_copy(num_, col_buffer + itr*num_*channels_ + ch*num_, top_data + ch*num_);
  }
  
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Backward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusion2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_1  = bottom[1]->cpu_data();
  
  Dtype* data_in   = data_in_.mutable_cpu_data();
  Dtype* weights   = weights_.mutable_cpu_data();
  Dtype* col_buffer= col_buffer_.mutable_cpu_data();
  Dtype* Normalization = normalization_.mutable_cpu_data();
  
  
  Dtype* Normalization2 = normalization2_.mutable_cpu_data();
  Dtype* alpha = alpha_.mutable_cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  if(propagate_down[0]){
  
  Weights_Reshape_Backward(bottom_1 , weights , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());

  for (int itr=(num_iter_-1);itr>=0;itr--){
  //=================================================================================================
  // Calculating E_y from E_z=top_diff
  //=================================================================================================    
  for(int i=0;i<num_;i++){
    //---------------------------------------------------------------------------------------------
    //Calculating N=sum(y,'across channels')
    //---------------------------------------------------------------------------------------------
    *(Normalization+i) = caffe_cpu_strided_dot(channels_, col_buffer+i, num_, ONEvector_ch.cpu_data() , 1);
    //---------------------------------------------------------------------------------------------
    //Calculating alpha = dot(E_z,y,'across channels') 
    //---------------------------------------------------------------------------------------------
    if(itr==(num_iter_-1))
      *(alpha+i) = caffe_cpu_strided_dot(channels_, col_buffer + itr*channels_*num_ + i, num_, top_diff+i , num_);
    else
      *(alpha+i) = caffe_cpu_strided_dot(channels_, col_buffer + itr*channels_*num_ + i, num_, bottom_diff+i , num_);      
  }    
  for(int ch=0;ch<channels_;ch++){
    //---------------------------------------------------------------------------------------------
    //Calculating Beta_ch = N*E_y_ch + alpha and Beta->col_buffer  
    //---------------------------------------------------------------------------------------------
    if(itr==(num_iter_-1))
      caffe_mul(num_, Normalization, top_diff+ch*num_, col_buffer + itr*channels_*num_ + ch*num_);
    else
      caffe_mul(num_, Normalization, bottom_diff+ch*num_, col_buffer + itr*channels_*num_ + ch*num_);    
    caffe_add(num_, alpha, col_buffer + itr*channels_*num_ + ch*num_, col_buffer + itr*channels_*num_ + ch*num_);
    //---------------------------------------------------------------------------------------------
    //Calculating E_y_ch = Beta_ch./(N.^2) 
    //---------------------------------------------------------------------------------------------
    caffe_mul(num_, Normalization, Normalization, Normalization2);
    caffe_div(num_, col_buffer + itr*channels_*num_ + ch*num_, Normalization2, col_buffer + itr*channels_*num_ + ch*num_);    
  }
  
  //=================================================================================================
  // Calculating E_x = A' * E_y
  //=================================================================================================
  
  
  for (int ch=0;ch<channels_;ch++){
      Data_Reshape_Backward(col_buffer + itr*channels_*num_ + ch*num_, data_in , im_height, im_width, seqX.cpu_data(), seqY.cpu_data());
      for(int i=0;i<num_;i++)
        *(bottom_diff + ch*num_+i) = caffe_cpu_dot(ne_, data_in+ne_*i, weights+ne_*i);
  }   
  }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef CPU_ONLY
STUB_GPU(MyDiffusion2Layer);
#endif

INSTANTIATE_CLASS(MyDiffusion2Layer);
REGISTER_LAYER_CLASS(MyDiffusion2);

}  // namespace caffe
