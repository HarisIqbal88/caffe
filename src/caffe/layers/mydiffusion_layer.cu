#include <vector>

#include "caffe/layers/mydiffusion_layer.hpp"
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
        //weights[ne_*pos+k]=bottom_1[ pos + k*num_ ];
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
////                          Calculate Weights Normalization
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void CalculateWeightsN(const Dtype* Input, Dtype* Normalization, int ne_, int num_){
  CUDA_KERNEL_LOOP(i, num_)
  {
    //---------------------------------------------------------------------------------------------
    //Calculating N=sum(y,'across channels')
    //---------------------------------------------------------------------------------------------
    Normalization[i] = 0;
    for(int ch=0;ch<ne_;ch++)
        Normalization[i] += Input[ne_*i + ch] + 0.0000000000001;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                        Normalize by dividing with 'Normalization' from above
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void NormalizeWeights(Dtype* weights, Dtype* Normalization, int ne_, int num_){
  CUDA_KERNEL_LOOP(i, ne_*num_)
  {
    //---------------------------------------------------------------------------------------------
    //Calculating w_new(m,n) = w(m,n)/N(m)  where m=1:num_,n=1:ne_
    //---------------------------------------------------------------------------------------------
    weights[i] /= Normalization[i/ne_]; 
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Weights Reshape Backwards
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void WeightsReshapeBackward(const Dtype* bottom_1, Dtype* weights,const Dtype* bottom_2, 
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
////                          tanh forward Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void TanhKernel_Forward(Dtype* x, Dtype* y, Dtype a, Dtype b, int count){
  CUDA_KERNEL_LOOP(i, count)
  {
      y[i] = b*tanh(a*x[i]);
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          tanh backward Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void TanhderivKernel_Backward(Dtype* data, const Dtype* top_grad, Dtype c, Dtype d, int count){
  CUDA_KERNEL_LOOP(i, count)
  {
      data[i] = d*top_grad[i]/( cosh(c*data[i])+1 );
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Forward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void MyDiffusionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<"Forward Pass starts: ";
  Dtype* top_data  = top[0]->mutable_gpu_data();
  const Dtype* bottom_0  = bottom[0]->gpu_data();
  const Dtype* bottom_1  = bottom[1]->gpu_data();
  const Dtype* bottom_2  = bottom[2]->gpu_data();
    
  Dtype* data_in   = data_in_.mutable_gpu_data();
  Dtype* weights   = weights_.mutable_gpu_data();
  Dtype* col_buffer= col_buffer_.mutable_gpu_data();
  
  Dtype* Normalization = normalization_.mutable_gpu_data();
  const int count = weights_.count();

  Dtype b = 1/tanh(a);
  
  WeightsReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	(bottom_1, weights , bottom_2, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
  CalculateWeightsN<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (weights, Normalization, ne_, num_);
  NormalizeWeights<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(weights, Normalization, ne_, num_);
  
  for(int itr=0;itr<num_iter_;itr++){
  for(int ch=0;ch<channels_;ch++){
    if(itr==0)   
      DataReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (bottom_0 + ch*num_, data_in + ch*num_*ne_, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
    else
      DataReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (top_data + ch*num_, data_in + ch*num_*ne_, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);      
  }
  DotProductKernel<<<CAFFE_GET_BLOCKS(count*channels_), CAFFE_CUDA_NUM_THREADS>>>( data_in, weights, count, channels_ );
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_*channels_,ne_, 1. , data_in, ONEvector_ne.gpu_data(), 0., col_buffer + itr*num_*channels_);
  
  if (tanh_nonlinearity && !( (itr-tanh_skip_start+1)%tanh_skip ))
    TanhKernel_Forward<<<CAFFE_GET_BLOCKS(num_*channels_), CAFFE_CUDA_NUM_THREADS>>>(col_buffer + itr*num_*channels_,top_data, a, b, num_*channels_);
  else
    caffe_gpu_scale(num_*channels_, (Dtype)1., col_buffer + itr*num_*channels_ , top_data) ;         
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
////                          Backward Pass
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void MyDiffusionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_1  = bottom[1]->gpu_data();
  const Dtype* bottom_2  = bottom[2]->gpu_data();
  
  Dtype* data_in   = data_in_b.mutable_gpu_data();
  Dtype* weights   = weights_b.mutable_gpu_data();
  Dtype* col_buffer= col_buffer_b.mutable_gpu_data();

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  Dtype* Normalization = normalization_.mutable_gpu_data();
  const int count = weights_.count();
  const Dtype d = 2*a / (tanh(a));
//  LOG(INFO)<<"Divided  by 4";
  const Dtype c = 2*a ; 

  WeightsReshapeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (bottom_1 , weights , bottom_2, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
//  caffe_gpu_memcpy(num_*5,weights, bottom_diff);     

  CalculateWeightsN<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (weights, Normalization, ne_, num_);
//  caffe_gpu_memcpy(num_,Normalization, bottom_diff + 6*num_);     

  NormalizeWeights<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(weights, Normalization, ne_, num_);
//  caffe_gpu_memcpy(num_*5,weights, bottom_diff + 8*num_);     

  
  if(propagate_down[0]){ 
  for(int itr=(num_iter_-1);itr>=0;itr--){
    if(tanh_nonlinearity && !( (itr-tanh_skip_start+1)%tanh_skip )){ 
        if(itr==(num_iter_-1))
            TanhderivKernel_Backward<<<CAFFE_GET_BLOCKS(num_*channels_), CAFFE_CUDA_NUM_THREADS>>>
		(col_buffer + itr*num_*channels_,top_diff, c, d, num_*channels_); // y,E_z -> d*E_z./{cosh(2*a*y)+1}
        else  
            TanhderivKernel_Backward<<<CAFFE_GET_BLOCKS(num_*channels_), CAFFE_CUDA_NUM_THREADS>>>
		(col_buffer + itr*num_*channels_,bottom_diff, c, d, num_*channels_);
    }
    else{
        if(itr==(num_iter_-1))
            caffe_gpu_memcpy(num_*channels_, top_diff, col_buffer + itr*num_*channels_);    
        else
            caffe_gpu_memcpy(num_*channels_, bottom_diff, col_buffer + itr*num_*channels_)  ;  
    }    
    for (int ch=0;ch<channels_;ch++){
        DataReshapeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
		(col_buffer + itr*channels_*num_ + ch*num_, data_in + ch*num_*ne_, im_height, im_width, seqX.gpu_data(), seqY.gpu_data(), num_,ne_);
    }
    DotProductKernel<<<CAFFE_GET_BLOCKS(count*channels_), CAFFE_CUDA_NUM_THREADS>>>( data_in, weights, count, channels_ );
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_*channels_,ne_, 1. , data_in, ONEvector_ne.gpu_data(), 0., bottom_diff);
  }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_LAYER_GPU_FUNCS(MyDiffusionLayer);


}  // namespace caffe
