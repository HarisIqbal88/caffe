#include <vector>

#include "caffe/layers/mydiffusion2_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Data Reshape Forward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void DataReshapeForward(const Dtype* bottom, Dtype* data_in, 
        int im_height, int im_width, const int* seqX, const int* seqY, int num_ , int ne_) {
  CUDA_KERNEL_LOOP(pos, num_*ne_)
  {
    int tmp_ = pos/ne_;
    int k = pos - ne_*tmp_; 
    int i = tmp_/im_width;
    int j = tmp_ - im_width*i;
    if (!( (j+seqX[k])<0 || (j+seqX[k])>=im_width || (i+seqY[k])<0 || (i+seqY[k])>=im_height ))
    {
        data_in[ne_*(i*im_width + j)+k]=bottom[ i*im_width + j + seqY[k]*im_width + seqX[k] ]; 
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Data Reshape Backward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void DataReshapeBackward(const Dtype* bottom, Dtype* data_in, 
        int im_height, int im_width, const int* seqX, const int* seqY, int num_ , int ne_) {
  CUDA_KERNEL_LOOP(pos, num_*ne_)
  {
    int tmp_ = pos/ne_;
    int k = pos - ne_*tmp_; 
    int i = tmp_/im_width;
    int j = tmp_ - im_width*i;
    if (!( (j-seqX[k])<0 || (j-seqX[k])>=im_width || (i-seqY[k])<0 || (i-seqY[k])>=im_height ))
    {
        data_in[ne_*( i*im_width + j )+k]=bottom[ (i*im_width + j) - (seqY[k]*im_width + seqX[k]) ]; 
    }
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Weights Reshape Forward
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void WeightsReshapeForward(const Dtype* bottom_1, Dtype* weights, const Dtype* bottom_2,
        int im_height, int im_width, const int* seqX, const int* seqY, int num_ , int ne_) {
  CUDA_KERNEL_LOOP(pos, num_*ne_)
  {
    int tmp_ = pos/ne_;
    int k = pos - ne_*tmp_; 
    int i = tmp_/im_width;
    int j = tmp_ - im_width*i;

    if (!( (j+seqX[k])<0 || (j+seqX[k])>=im_width || (i+seqY[k])<0 || (i+seqY[k])>=im_height ))
    { 
        if (bottom_2[i*im_width + j]==0)
            weights[ne_*(i*im_width + j)+k]=bottom_1[ i*im_width + j + k*num_ ];
        else if (k==0)
            weights[ne_*(i*im_width + j)]=1;
        else if (k>0)
            weights[ne_*(i*im_width + j)+k]=0;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Weights Reshape Backwards
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void WeightsReshapeBackward(const Dtype* bottom_1, Dtype* weights, const Dtype* bottom_2,
    int im_height, int im_width, const int* seqX, const int* seqY, int num_ , int ne_) {
  CUDA_KERNEL_LOOP(pos, num_*ne_)
  {
    int tmp_ = pos/ne_;
    int k = pos - ne_*tmp_; 
    int i = tmp_/im_width;
    int j = tmp_ - im_width*i;

    if (!( (j-seqX[k])<0 || (j-seqX[k])>=im_width || (i-seqY[k])<0 || (i-seqY[k])>=im_height ))
    { 
        if (bottom_2[i*im_width + j]==0)
            weights[ne_*(i*im_width + j)+k]=bottom_1[ (i*im_width + j) - ( seqY[k]*im_width + seqX[k] ) +k*num_];
        else if (k==0)
            weights[ne_*(i*im_width + j)]=1;
        else if (k>0)
            weights[ne_*(i*im_width + j)+k]=0;
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Dot Product Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void DotProductKernel(Dtype* data_in, Dtype* weights, int count, int channels_){
  CUDA_KERNEL_LOOP(i, count*channels_)
  {
      data_in[i] *= weights[i%count];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Calculate Normalization
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void CalculateN(const Dtype* col_buffer, Dtype* Normalization, int channels_, int num_){
  CUDA_KERNEL_LOOP(i, num_)
  {
    //---------------------------------------------------------------------------------------------
    //Calculating N=sum(y,'across channels')
    //---------------------------------------------------------------------------------------------
    for(int ch=0;ch<channels_;ch++)
        Normalization[i] += col_buffer[i + ch*num_];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Forward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void MyDiffusion2Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Dtype* top_data  = top[0]->mutable_gpu_data();
  const Dtype* bottom_0  = bottom[0]->gpu_data();
  const Dtype* bottom_1  = bottom[1]->gpu_data();
  const Dtype* bottom_2  = bottom[2]->gpu_data();
    
  Dtype* data_in   = data_in_.mutable_gpu_data();
  Dtype* weights   = weights_.mutable_gpu_data();
  Dtype* col_buffer= col_buffer_.mutable_gpu_data();
  Dtype* Normalization = normalization_.mutable_gpu_data();
  
  const int count = weights_.count();

  WeightsReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_1 , weights, bottom_2, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
  
  for(int itr=0;itr<num_iter_;itr++){
  LOG(INFO)<<"iteration #: "<< itr;
  for(int ch=0;ch<channels_;ch++){
    if(itr==0)   
      DataReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_0 + ch*num_, data_in + ch*num_*ne_, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
    else
      DataReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_data + ch*num_, data_in + ch*num_*ne_, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);      
  } 
  DotProductKernel<<<CAFFE_GET_BLOCKS(count*channels_), CAFFE_CUDA_NUM_THREADS>>>( data_in, weights, count, channels_ );
  LOG(INFO)<<"iteration #: "<< itr << "  Elwise dotproduct kernel done";
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_*channels_,ne_, 1. , data_in, ONEvector_ne.gpu_data(), 0., col_buffer + itr*num_*channels_);

  CalculateN<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(col_buffer + itr*channels_*num_ , Normalization, channels_, num_);
  
  for(int ch=0;ch<channels_;ch++)
    caffe_gpu_div(num_, col_buffer + itr*channels_*num_ + ch*num_, Normalization, top_data + ch*num_);
  
  } 
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Calculate Normalization and Alpha Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void CalculateNandAlpha(const Dtype* col_buffer, const Dtype* top_grad , const Dtype* ONEvector_ch, Dtype* Normalization, Dtype* alpha, int channels_, int num_){
  CUDA_KERNEL_LOOP(i, num_)
  {
    Normalization[i] = 0;
    alpha[i] = 0;
    //---------------------------------------------------------------------------------------------
    //Calculating N=sum(y,'across channels')
    //---------------------------------------------------------------------------------------------
    for(int ch=0;ch<channels_;ch++)
        Normalization[i] += col_buffer[i + ch*num_];
    //caffe_gpu_strided_dot(channels_, col_buffer+i, num_, ONEvector_ch , 1, Normalization+i);
    //---------------------------------------------------------------------------------------------
    //Calculating alpha = dot(E_z,y,'across channels') 
    //---------------------------------------------------------------------------------------------
    for(int ch=0;ch<channels_;ch++)
        alpha[i] += col_buffer[i + ch*num_]*top_grad[i + ch*num_];
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Backward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusion2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_1  = bottom[1]->gpu_data();
  const Dtype* bottom_2  = bottom[2]->gpu_data();
  
  Dtype* data_in   = data_in_b.mutable_gpu_data();
  Dtype* weights   = weights_b.mutable_gpu_data();
  Dtype* col_buffer= col_buffer_b.mutable_gpu_data();
  Dtype* Normalization = normalization_.mutable_gpu_data();
  
  Dtype* Normalization2 = normalization2_.mutable_gpu_data();
  Dtype* alpha = alpha_.mutable_gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  const int count = weights_.count();
  
  if(propagate_down[0]){
  
  WeightsReshapeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_1 , weights, bottom_2, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);

  for (int itr=(num_iter_-1);itr>=0;itr--){
  //=================================================================================================
  // Calculating E_y from E_z=top_diff
  //=================================================================================================    
  if(itr==(num_iter_-1))
    CalculateNandAlpha<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(col_buffer + itr*channels_*num_,  top_diff , ONEvector_ch.cpu_data(), Normalization, alpha, channels_, num_);
  else
    CalculateNandAlpha<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(col_buffer + itr*channels_*num_,  bottom_diff , ONEvector_ch.cpu_data(), Normalization, alpha, channels_, num_);
     
  for(int ch=0;ch<channels_;ch++){
    //---------------------------------------------------------------------------------------------
    //Calculating Beta_ch = N*E_y_ch + alpha and Beta->col_buffer  
    //---------------------------------------------------------------------------------------------
    if(itr==(num_iter_-1))
      caffe_gpu_mul(num_, Normalization, top_diff+ch*num_, col_buffer + itr*channels_*num_ + ch*num_)  ;
    else
      caffe_gpu_mul(num_, Normalization, bottom_diff+ch*num_, col_buffer + itr*channels_*num_ + ch*num_);    
    caffe_gpu_add(num_, alpha, col_buffer + itr*channels_*num_ + ch*num_, col_buffer + itr*channels_*num_ + ch*num_);
    //---------------------------------------------------------------------------------------------
    //Calculating E_y_ch = Beta_ch./(N.^2) 
    //---------------------------------------------------------------------------------------------
    caffe_gpu_mul(num_, Normalization, Normalization, Normalization2);
    caffe_gpu_div(num_, col_buffer + itr*channels_*num_ + ch*num_, Normalization2, col_buffer + itr*channels_*num_ + ch*num_);    
  }
  
  //=================================================================================================
  // Calculating E_x = A' * E_y
  //=================================================================================================
  for (int ch=0;ch<channels_;ch++)
    DataReshapeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(col_buffer + itr*channels_*num_ + ch*num_, data_in + ch*num_*ne_, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
  
  DotProductKernel<<<CAFFE_GET_BLOCKS(count*channels_), CAFFE_CUDA_NUM_THREADS>>>( data_in, weights, count, channels_ );
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_*channels_,ne_, 1. , data_in, ONEvector_ne.gpu_data(), 0., bottom_diff); 
  
  }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_LAYER_GPU_FUNCS(MyDiffusion2Layer);


}  // namespace caffe
