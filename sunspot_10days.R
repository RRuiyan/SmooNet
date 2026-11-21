library(forecast)
library(fracdiff)
library(rugarch)

library(fda)
library(Rcpp)
library(RcppEigen) 
#library(CollocInfer)  
source("dynamic.FDE.R")
load("sunspots_1849_10days.RData") 

Y=sunspot10d$Y0

#####################################

for(rep in 1:100)
{
  for(forward_predict_time in c(3, 6))
  {
    delta_t=1/36 
    forward_steps=floor(forward_predict_time/delta_t)  
    N=floor(100/delta_t)
    
    errors=rep(0,9)      
    times=rep(0,9)
    t0=proc.time()[3] 
    M=length(Y)
    N_start=sample(1:(M-N- forward_steps),1)
    Y.all.train=Y[N_start+(1:N)]
    Y.test=Y[N_start+((N+1):(N+forward_steps))]
    
    fit_model=model.fit(Y=Y.all.train, delta_t=delta_t, gap=3, forward_predict_time=forward_predict_time, max_iters=90)
    run.times=proc.time()[3]-t0
    
    train=fit_model$opt_train
    depend_prev_time=train[[4]]
    n.s=floor(depend_prev_time/delta_t)
    
    Y.test.prev=Y[N_start+((N-n.s+1):N)]
    
    errors[1]=predict.dynamic_FDE(Y.test, Y.test.prev, train)$error 
    times[1]=times_aic=(proc.time()[3]-t0) 
    

    results=c(errors) 
    file.root<- paste("sunspot_10days_allmethods_forward_predict_time=", forward_predict_time,  sep="")
    write.table(t(results),file=paste(file.root, ".error.txt", sep=""), 
                row.names=FALSE, col.names=FALSE, append=TRUE) 
    
  }
}
