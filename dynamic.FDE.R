
library(fda)
library(Rcpp)
library(RcppEigen)

sourceCpp("c_dynamic_FDE.cpp")


#########################################
###################################################################
#######################################################################
#' @export
model.fit=function(Y, train.prop=2/3, delta_t, s.n.basis=30, u.n.basis=30, 
                   n.u=50, gap=2, forward_predict_time, upper.comp=5, max_iters=100)
{
  depend_prev_time_list=list() 
  error_list=list()
  train.list=list()
  k=1
  depend_prev_time_list[[k]]=2*forward_predict_time
  train=select.smooth.params(Y=Y, delta_t=delta_t, gap=gap, depend_prev_time=depend_prev_time_list[[k]], forward_predict_time=forward_predict_time, max_iters=90)
  train.list[[k]]=train 
  error_list[[k]]=train[[1]][[1]] 
  k=2
  depend_prev_time_list[[k]]=depend_prev_time_list[[k-1]]/2
  train=select.smooth.params(Y=Y.all.train, delta_t=delta_t, gap=gap, depend_prev_time=depend_prev_time_list[[k]], forward_predict_time=forward_predict_time, max_iters=90)
  train.list[[k]]=train 
  error_list[[k]]=train[[1]][[1]]
  if(error_list[[2]]<error_list[[1]])
  { 
    depend_prev_time_list[[3]]=depend_prev_time_list[[2]]/2
  }else
  {
    depend_prev_time_list[[3]]=depend_prev_time_list[[1]]*2
  }
  k=3
  train=select.smooth.params(Y=Y.all.train, delta_t=delta_t, gap=gap, depend_prev_time=depend_prev_time_list[[k]], forward_predict_time=forward_predict_time, max_iters=90)
  train.list[[k]]=train 
  error_list[[k]]=train[[1]][[1]]
  min.error=1e30
  for(k in 1:3)
  {
    if(error_list[[k]]<min.error)
    {
      min.error=error_list[[k]]
      min.index=k
    }
  }
  opt_train=train.list[[min.index]] 
  
  return(list(error_list=error_list, train.list=train.list, depend_prev_time_list=depend_prev_time_list,
              opt_train=opt_train, min.index=min.index))
}
###################################################################
#######################################################################
#' @export
select.smooth.params=function(Y, train.prop=2/3, delta_t, depend_prev_time, s.n.basis=30, u.n.basis=30, 
                              n.u=50, gap=2, forward_predict_time, upper.comp=5, max_iters=100)
{
  m=floor(length(Y)*train.prop)
  Y.train=Y[1:m]
  Y.valid.all=Y
  
  h=delta_t
  n.s=floor(depend_prev_time/h)
  forward_steps=floor(forward_predict_time/h)
  bspline.s.obj=create.bspline.basis(c(0,1), s.n.basis, 4)
  
  Psi=eval.basis(seq(0, 1, length.out = n.s), bspline.s.obj)
  
  Y.input=Y.train[(n.s+1):length(Y.train)]   
  bspline.u.obj=create.bspline.basis(c(0,1), u.n.basis, 4) 
  Phi=eval.basis(seq(0, 1, length.out = n.u), bspline.u.obj) 
  
  J.s0=getbasispenalty(bspline.s.obj, 0) 
  J.u0=getbasispenalty(bspline.u.obj, 0)
  J.s2=getbasispenalty(bspline.s.obj, 2) 
  J.u2=getbasispenalty(bspline.u.obj, 2)
  nus=c(1e-9, 5e-9, 1e-8,5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
  
  
  
  x.param.list=list() 
  x.param.list[[1]]=Phi  
  x.param.list[[2]]=Psi  
  x.param.list[[3]]=J.s0 
  x.param.list[[4]]=J.u0  
  x.param.list[[5]]=J.s2 
  x.param.list[[6]]=J.u2   
  x.param.list[[7]]=nus
  x.param.list[[8]]=c(1e-5,  1e-5)
  
  seg_end_index_C=t(sapply(0:((length(Y.input)-forward_steps)/gap-1), function(k){k*gap+c(1,forward_steps)-1}))
  seg_N=sapply(1:dim(seg_end_index_C)[1], function(k){seg_end_index_C[k,2]-seg_end_index_C[k,1]+1})
  delta_Y=sapply(1:dim(seg_end_index_C)[1], function(k){Y.input[seg_end_index_C[k,2]+1]-Y.input[seg_end_index_C[k,1]+1]})
  delta_t_vec=sapply(1:dim(seg_end_index_C)[1], function(k){(seg_N[k]-1)*h})
  
  mu_vec=(delta_t_vec/seg_N)/n.u
  
  x.param.list[[9]]=seg_end_index_C
  x.param.list[[10]]=seg_N
  x.param.list[[11]]=delta_Y
  x.param.list[[12]]=mu_vec
  
  
  Y.mat=sapply((n.s+1):length(Y.train), function(k){(Y.train[(k-n.s+1):k])})
  
  ZZ.ls=c_smoothing_multiple(Y.mat, Psi,  J.s2, nus)
  ZZ=t(J.s0%*%ZZ.ls$W)
  x.param.list[[13]]=ZZ
  
  m=length(Y.train)+1
  N=(length(Y.valid.all)-m-forward_steps)
  
  
  Y.CV.prev.smooth=NULL
  Y.CV.true=NULL
  for(k in 0:N)
  { 
    Y0=Y.valid.all[(k+m-n.s):(k+m-1)]  #m: length of training
    fit=c_smoothing_single(Y0,  Psi,  J.s2, nus) 
    Y.CV.prev.smooth=cbind(Y.CV.prev.smooth, rev(Psi%*%fit$w)) 
    Y.CV.true=cbind(Y.CV.true, Y.valid.all[(k+m):(k+m+forward_steps-1)])  
  }
  
  x.param.list[[14]]=Y.CV.prev.smooth
  x.param.list[[15]]=Y.CV.true  
  
  lambda_list=list()
  lambda_list[[1]]=c(1e-9, 1e-9)
  lambda_list[[2]]=c(1e-5, 1e-5)
  min.err=1e30
  for(k in 1:(length(lambda_list)))
  {
    x.param.list[[8]]=lambda_list[[k]]
    fit_tmp=c_select_tuning_parameter(x.param.list, max_iters, forward_steps, delta_t, upper.comp)
    error=fit_tmp[[1]]
    if(error<min.err)
    {
      min.err=error
      fit=fit_tmp
    }
  }
  
  fit_list=list()
  fit_list[[1]]=fit
  fit_list[[2]]=x.param.list
  fit_list[[3]]=delta_t
  fit_list[[4]]=depend_prev_time
  fit_list[[5]]=forward_predict_time
  
  
  return(fit_list)
}

################################################################################
predict.dynamic_FDE=function(Y.test, Y.test.prev, fit_list)
{
  x.param.list=fit_list[[2]]
  delta_t=fit_list[[3]]
  depend_prev_time=fit_list[[4]]
  forward_predict_time=fit_list[[5]]
  Phi=x.param.list[[1]]
  Psi=x.param.list[[2]]
  J.s2=x.param.list[[5]]
  fit=fit_list[[1]]
  D=fit[[4]][[fit$optParms[[1]]-1]]
  
  ahat=D$a
  chat=D$c
  Bhat=D$B
  Vhat=D$V 
  
  h=delta_t
  n.s=floor(depend_prev_time/h)
  forward_steps=floor(forward_predict_time/h)
  nus=x.param.list[[7]] 
  fit.smooth=c_smoothing_single(Y.test.prev,  Psi,  J.s2, nus) 
  Y.prev.smooth=rev(Psi%*%fit.smooth$w) 
  
  Yc=c_predict(Y.prev.smooth, forward_steps, delta_t, Phi, Psi,  ahat,  Bhat,  Vhat, chat) 
  if(is.null(Y.test)){
    error=NULL
  }else{
    error=mean((Yc[[1]][n.s+(1:forward_steps)]-Y.test)^2)  
  }
#  error=mean((Yc[[1]][n.s+(1:forward_steps)]-Y.test)^2)  
  #return(error)
  return(list(error=error, forecast=Yc[[1]][n.s+(1:forward_steps)]))
}
