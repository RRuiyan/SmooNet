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
     
    
    #neural network NNAR(p,k)
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
    
    
    period=400
    
    #sARIMA, period=400
    t0=proc.time()[3]
    auto.fit.s=auto.arima(ts(Y.all.train, frequency=period), ic ="aic") #order: p,q,P,Q,period,d,D
    auto.pred.s=forecast(auto.fit.s, h=forward_steps)$mean 
    errors[5]=mean((auto.pred.s-Y.test)^2)
    times[5]=(proc.time()[3]-t0) 
    
    
    
    #dynamic harmonic regression, period=400
    #harmonic regression terms force the seasonal patterns to repeat periodically without changing
    t0=proc.time()[3]
    t.cur=N_start+(1:N)
    period=400
    h=forward_steps #200
    x.fourier.train=cbind(sin(2*pi*t.cur/period), cos(2*pi*t.cur/period))
    x.fourier.test=cbind(sin(2*pi*(tail(t.cur,1)+1:h)/period), cos(2*pi*(tail(t.cur,1)+1:h)/period))
    dynHarmn.fit=auto.arima(Y.all.train, seasonal=FALSE, xreg=x.fourier.train)         
    dynHarmn.fst=forecast(dynHarmn.fit, xreg=x.fourier.test) 
    errors[6]=errors.dyn=mean((dynHarmn.fst$mean-Y.test)^2) 
    times_aic.s=(proc.time()[3]-t0) 
    times[6]=proc.time()[3]-t0
    
    dynHarmn.fit %>% 
      forecast(xreg=x.fourier.test) %>% 
      autoplot()
    
    
    
    #STL 
    t0=proc.time()[3]
    mstl.res=mstl(Y.all.train)
    mstl.fst=stlf(ts(Y.all.train, frequency=400), h=forward_steps)
    errors[7]=errors.mstl=mean((mstl.fst$mean-Y.test)^2) 
    ts(Y.all.train, frequency=400) %>% stlf(h=forward_steps) %>% 
      autoplot()
    times[7]=proc.time()[3]-t0
    
    #TBATS
    #seasonality is allowed to change slowly over time in a TBATS model
    t0=proc.time()[3]
    tbats.res=tbats(ts(Y.all.train, frequency=400))
    tbats.fst=forecast(tbats.res, h=forward_steps)
    errors[8]=errors.mstl=mean((tbats.fst$mean-Y.test)^2) 
    times[8]=proc.time()[3]-t0
    plot(tbats.fst)
    
    
    #seasonal neural network NNAR(p,k), 400
    t0=proc.time()[3]
    m=floor(length(Y.all.train)*2/3)
    Y.train=Y.all.train[1:m]
    Y.valid=Y.all.train[(m+1):length(Y.all.train)]
    decay.set= 0.1*c(0:9)
    valid.err=rep(0, length(decay.set))
    for(m in 1:length(decay.set))
    {
      nnar.res=nnetar(ts(Y.all.train, frequency=400),decay=decay.set[m])
      nnar.fst=forecast(nnar.res, h=length(Y.valid))
      valid.err[m]=mean((nnar.fst$mean-Y.valid)^2) 
      print(c(m, valid.err[m]))
    }
    decay.opt1= decay.set[which.min(valid.err)] 
    nnar.res=nnetar(ts(Y.all.train, frequency=400), decay=decay.opt1)
    nnar.fst=forecast(nnar.res, h=forward_steps)
    errors[9]=errors.nnar=mean((nnar.fst$mean-Y.test)^2) 
    times[9]=proc.time()[3]-t0 
    

    
    results=c(errors) 
    file.root<- paste("sunspot_10days_allmethods_forward_predict_time=", forward_predict_time,  sep="")
    write.table(t(results),file=paste(file.root, ".error.txt", sep=""), 
                row.names=FALSE, col.names=FALSE, append=TRUE) 
    
  }
}
