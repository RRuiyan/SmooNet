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
      
      return(sum(0.1*s^0.5*(1-x)*x^2)*h)
    }
    
    h=0.01
    interval.low=-5+h
    tot.len=floor(200/h)  
    
    t.prev=seq(interval.low,0,by=h)
    delta.length=length(t.prev)
    y.prev=10*sin(t.prev)
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
    
    #ARIMA.short
    t0=proc.time()[3]
    auto.fit=auto.arima(Y.all.train, ic ="aic") #order: p,q,P,Q,period,d,D
    auto.pred=forecast(auto.fit, h=forward_steps)$mean 
    errors[2]=errors.auto.aic=mean((auto.pred-Y.test)^2)  
    times[2]=times_aic=(proc.time()[3]-t0)  
    
    #ARIMA.long
    t0=proc.time()[3] 
    print("starting time=")
    print(Sys.time())
    f43=rugarch::autoarfima(Y.all.train, ar.max=3, ma.max=3, criterion = "AIC", 
                            method = "full",arfima = NULL, include.mean = NULL,
                            fit.control = list(scale = 1), solver="nlminb") 
    pred43=arfimaforecast(f43$fit, n.ahead=forward_steps)
    #  (mean((pred33@forecast[[5]]-Y.test)^2))
    errors[3]=errors43=mean((pred43@forecast[[5]]-Y.test)^2) 
    times[3]=times43=(proc.time()[3]-t0)  
    pred43.values=pred43@forecast[[5]]
    lines(((length(Y.all.train)+1):K), pred43@forecast[[5]], col='green')
    
    #neural network NNAR(p,k)
    t0=proc.time()[3]
    t0=proc.time()[3]
    m=floor(length(Y.all.train)*2/3)
    Y.train=Y.all.train[1:m]
    Y.valid=Y.all.train[(m+1):length(Y.all.train)]
    decay.set= 0.1*c(0:9)
    valid.err=rep(0, length(decay.set))
    for(m in 1:length(decay.set))
    {
      nnar.res=nnetar(Y.train,decay=decay.set[m])
      nnar.fst=forecast(nnar.res, h=length(Y.valid))
      valid.err[m]=mean((nnar.fst$mean-Y.valid)^2) 
      print(c(m, valid.err[m]))
    }
    
    decay.opt= decay.set[which.min(valid.err)] 
    
    nnar.res=nnetar(Y.all.train, decay=decay.opt)
    nnar.fst=forecast(nnar.res, h=forward_steps)
    errors[4]=errors.nnar=mean((nnar.fst$mean-Y.test)^2) 
    times[4]=proc.time()[3]-t0 
    
    
    results=c(errors) 
    file.root<- paste("sim.3_forward_predict_time=", forward_predict_time,  sep="")
    write.table(t(results),file=paste(file.root, ".error.txt", sep=""), 
                row.names=FALSE, col.names=FALSE, append=TRUE) 
    
  }
}
