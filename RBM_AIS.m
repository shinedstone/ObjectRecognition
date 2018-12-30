function logZZ_est = RBM_AIS(vishid,hidbiases,visbiases,numruns,beta)

% close all
% figure('Position',[100,600,500,200]);
% figure(2)
% hold on
% xlabel('beta','fontsize',14)
% ylabel('Variance of log weights','fontsize',14)

[numdims numhids]=size(vishid);
if(nargin>5)
%%% Initialize biases of the base rate model by ML %%%%%%%%%%%%%%%%%%%%%%%
    base_rate
    visbiases_base = log_base_rate';
else
    visbiases_base = 0*visbiases;
end

numcases = numruns;

%%%%%%%%%% RUN AIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
visbias_base = repmat(visbiases_base,numcases,1);
hidbias = repmat(hidbiases,numcases,1);
visbias = repmat(visbiases,numcases,1);

%%%% Sample from the base-rate model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logww = zeros(numcases,1);
negdata = repmat(1./(1+exp(-visbiases_base)),numcases,1);
negdata = negdata > rand(numcases,numdims);
logww  =  logww - (negdata*visbiases_base' + numhids*log(2));

Wh = negdata*vishid + hidbias;
Bv_base = negdata*visbiases_base';
Bv = negdata*visbiases';
tt=1;

%%% The CORE of an AIS RUN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for bb = beta(2:end-1);
%     fprintf(1,'beta=%d\r',bb);
    tt = tt+1;
    
    expWh = exp(bb*Wh);
    logww  =  logww + (1-bb)*Bv_base + bb*Bv + sum(log(1+expWh),2);
    
    poshidprobs = expWh./(1 + expWh);
    poshidstates = poshidprobs > rand(numcases,numhids);
    
    negdata = 1./(1 + exp(-(1-bb)*visbias_base - bb*(poshidstates*vishid' + visbias)));
    negdata = negdata > rand(numcases,numdims);
    
%     if rem(tt,500)==0
%         figure(1)
%         mnistdisp(negdata(1:10,:)');
%         
%         figure(2)
%         plot(tt/length(beta),var(logww(:)),'b*')
%         hold on
%         drawnow;
%     end
    
    Wh      = negdata*vishid + hidbias;
    Bv_base = negdata*visbiases_base';
    Bv      = negdata*visbiases';
    
    expWh = exp(bb*Wh);
    logww  =  logww - ((1-bb)*Bv_base + bb*Bv + sum(log(1+expWh),2));
    
end

expWh = exp(Wh);
logww  = logww +  negdata*visbiases' + sum(log(1+expWh),2);

r_AIS = logsum(logww(:)) -  log(numcases);

logZZ_base = sum(log(1+exp(visbiases_base))) + (numhids)*log(2);
logZZ_est = r_AIS + logZZ_base;