library(forecast)
library(fracdiff)
library(rugarch)

library(fda)
library(Rcpp)
library(RcppEigen) 
source("dynamic.FDE.R")


for(rep in 1:100)
{
  for(forward_predict_time in c(3, 6))
  {
    
    #######################################
    ###################data simulation##############
    ########################################
    f=function(x, h)
    { 
      s=seq(0,1,length.out = length(x))
      y=0.4*sum(sin(x*s)*cos(x))*h 
      return(y/sqrt(1+y^2))
    }
    
    
    h=0.01
    interval.low=-10+h
    tot.len=floor(200/h)  
    
    
    t.prev=seq(interval.low,0,by=h)
    delta.length=length(t.prev)
    y.prev=5*cos(2*pi*t.prev) 
    dy.prev=rep(0, length(t.prev))
    VV=rbind(t.prev, y.prev, dy.prev)
    h=t.prev[2]-t.prev[1]
    
    
    
    
    y.prev=VV[2, (dim(VV)[2]-delta.length+1):dim(VV)[2]]
    y.cur=VV[2, dim(VV)[2]]
    t.cur=VV[1, dim(VV)[2]]
    t.new=t.cur+h
    dy.cur=f(y.prev, h)
    y.new=y.cur+h*dy.cur
    VV[3,dim(VV)[2]]=dy.cur
    VV=cbind(VV, c(t.new, y.new,0))
    
    y.prev=VV[2, (dim(VV)[2]-delta.length+1):dim(VV)[2]]
    y.cur=VV[2, dim(VV)[2]]
    t.cur=VV[1, dim(VV)[2]]
    t.new=t.cur+h
    dy.cur=f(y.prev, h)
    dy_1=VV[3,dim(VV)[2]-1]
    y.new=y.cur+(3*dy.cur-dy_1)*h/2
    VV[3,dim(VV)[2]]=dy.cur
    VV=cbind(VV, c(t.new, y.new, 0))
    
    y.prev=VV[2, (dim(VV)[2]-delta.length+1):dim(VV)[2]]
    y.cur=VV[2, dim(VV)[2]]
    t.cur=VV[1, dim(VV)[2]]
    t.new=t.cur+h
    dy.cur=f(y.prev, h)
    dy_1=VV[3,dim(VV)[2]-1]
    dy_2=VV[3,dim(VV)[2]-2]
    y.new=y.cur+(23*dy.cur-16*dy_1+5*dy_2)*h/12
    VV[3,dim(VV)[2]]=dy.cur
    VV=cbind(VV, c(t.new, y.new, 0))
    skip=floor(50/h)
    K.total=tot.len+skip
    for(k in 1:K.total)
    {
      y.prev=VV[2, (dim(VV)[2]-delta.length+1):dim(VV)[2]]
      y.cur=VV[2, dim(VV)[2]]
      t.cur=VV[1, dim(VV)[2]]
      t.new=t.cur+h
      dy.cur=f(y.prev, h)
      dy_1=VV[3,dim(VV)[2]-1]
      dy_2=VV[3,dim(VV)[2]-2]
      dy_3=VV[3,dim(VV)[2]-3]
      y.new=y.cur+(55*dy.cur-59*dy_1+37*dy_2-9*dy_3)*h/24
      VV[3,dim(VV)[2]]=dy.cur
      VV=cbind(VV, c(t.new, y.new, 0))
    }
    
    
    sigma=sqrt(0.01)
    t.seq=VV[1, skip:K.total] 
    
    E=sigma*rnorm(length(t.seq))
    S=VV[2, skip:K.total]
    
    m=min(S)
    M=max(S)
    S=2*(S-m)/(M-m)-1
    Y0=S+E 
    #####################################
    delta_t=0.05 
    
    k0=floor(delta_t/0.01)
    K= floor(length(Y0)/k0) 
    Y=Y0[(1:K)*k0]  
    S.sp=S[(1:K)*k0] 
    ##########################################################################
    ############### model fitting #####################################
    ##########################################################################
    
    errors=rep(0,4)      
    times=rep(0,4)
    forward_steps=floor(forward_predict_time/delta_t) 
    N=floor(100/delta_t) 
    M=length(Y)
    N_start=sample(1:(M-N- forward_steps),1)  
    Y.all.train=Y[N_start+(1:N)]
    Y.test=Y[N_start+((N+1):(N+forward_steps))] 
    print(N_start)
    K=length(c(Y.all.train, Y.test)) 
    ############### 
    # new FDE method
    t0=proc.time()[3] 
    fit_model=model.fit(Y=Y.all.train, delta_t=delta_t,   forward_predict_time=forward_predict_time)
    train=fit_model$opt_train
    depend_prev_time=train[[4]]
    
    n.s=floor(depend_prev_time/delta_t)
    
    Y.test.prev=Y[N_start+((N-n.s+1):N)]
    pred.res=predict.dynamic_FDE(Y.test, Y.test.prev, train)
    errors[1]=pred.res$error  
    times[1]=times_aic=(proc.time()[3]-t0) 
    
    plot(1:K, c(Y.all.train, Y.test), main=paste(rep, forward_predict_time, errors[1], sep=","))
    lines(((length(Y.all.train)+1):K), pred.res$forecast, col='red')
    

    results=c(errors) 
    file.root<- paste("sim.1_forward_predict_time=", forward_predict_time,  sep="")
    write.table(t(results),file=paste(file.root, ".error.txt", sep=""), 
                row.names=FALSE, col.names=FALSE, append=TRUE) 
    
  }
}
