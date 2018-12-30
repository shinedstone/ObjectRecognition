%% Initializing symmetric weights and biases.
vishid = vishid0;
hidbiases  = hidbiases0;
visbiases  = visbiases0;

poshidprobs = zeros(numcases,numhid);
posprods    = zeros(numdims,numhid);
negprods    = zeros(numdims,numhid);
vishidinc  = zeros(numdims,numhid);
hidbiasinc = zeros(1,numhid);
visbiasinc = zeros(1,numdims);
batchposhidprobs = zeros(numcases,numhid,numbatches);
Error = zeros(1,maxepoch);
Logprob = [];
Logprob2 = [];

temperature = 0.9:0.0033:1;
nchain = size(temperature,2);
negdata = zeros(nchain,numdims);
for i = 1 : 500
    neghidprobs = 1./(1 + exp(-(negdata.*repmat(temperature',1,numdims))*vishid-temperature'*hidbiases));
    neghidstates = neghidprobs > rand(nchain,numhid);
    negdata=1./(1 + exp(-(neghidstates.*repmat(temperature',1,numhid))*vishid'-temperature'*visbiases));
    negdata = negdata > rand(nchain,numdims);
end

for epoch = 1:maxepoch
    fprintf(1,'tmcmc - epoch %d\r',epoch);
    errsum = 0;
    epsilonw      = 0.01/(1+epoch/3000);
    epsilonvb     = 0.01/(1+epoch/3000);
    epsilonhb     = 0.01/(1+epoch/3000);
    for batch = 1:numbatches
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        poshidprobs = 1./(1 + exp( - data*vishid - repmat(hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods=data'*poshidprobs;
        poshidact=sum(poshidprobs);
        posvisact=sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i = 1 : nchain-1
            E11 = -(negdata(i,:)*(visbiases*temperature(i))'+sum(log(1+exp(hidbiases*temperature(i)+negdata(i,:)*(vishid*temperature(i)))),2));
            E11s = -(negdata(i,:)*(visbiases*temperature(i+1))'+sum(log(1+exp(hidbiases*temperature(i+1)+negdata(i,:)*(vishid*temperature(i+1)))),2));
            E22 = -(negdata(i+1,:)*(visbiases*temperature(i+1))'+sum(log(1+exp(hidbiases*temperature(i+1)+negdata(i+1,:)*(vishid*temperature(i+1)))),2));
            E22s = -(negdata(i+1,:)*(visbiases*temperature(i))'+sum(log(1+exp(hidbiases*temperature(i)+negdata(i+1,:)*(vishid*temperature(i)))),2));
            
            swap_prob = min( 1 , exp(E11 - E11s + E22 - E22s) );
            
            swapping_particles = binornd(1, swap_prob);
            staying_particles = 1 - swapping_particles;
            
            swp_p_visible = repmat(swapping_particles, [1 numdims]);
            sty_p_visible = repmat(staying_particles, [1 numdims]);
            
            v1t1 = negdata(i,:).*swp_p_visible;
            v1t2 = negdata(i+1,:).*swp_p_visible;
            
            negdata(i,:) = negdata(i,:).*sty_p_visible;
            negdata(i+1,:) = negdata(i+1,:).*sty_p_visible;
            
            negdata(i,:) = negdata(i,:)+v1t2;
            negdata(i+1,:) = negdata(i+1,:)+v1t1;
        end
        
        neghidprobs = 1./(1+exp(-negdata(end,:)*vishid-hidbiases));
        
        negprods=negdata(end,:)'*neghidprobs;
        neghidact=neghidprobs;
        negvisact=negdata(end,:);
        
        neghidprobs = 1./(1+exp(-(negdata.*repmat(temperature',1,numdims))*vishid-temperature'*hidbiases));
        neghidstates = neghidprobs > rand(nchain,numhid);
        
        negdata = 1./(1 + exp(-(neghidstates.*repmat(temperature',1,numhid))*vishid'-temperature'*visbiases));
        negdata = negdata > rand(nchain,numdims);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = epsilonw*(posprods/numcases-negprods);
        visbiasinc = epsilonvb*(posvisact/numcases-negvisact);
        hidbiasinc = epsilonhb*(poshidact/numcases-neghidact);
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        
        errsum = errsum + sum((data-(1./(1+exp(-poshidstates*vishid'-repmat(visbiases,numcases,1)))>rand(numcases,numdims))).^2,2);
    end
    Error(epoch) = mean(errsum/numbatches);
    if rem(log2(epoch),1) == 0
        logZZ_est = 0;
        for i = 1 : 10
            logZZ_est = logZZ_est + RBM_AIS(vishid,hidbiases,visbiases,numruns,beta);
        end
        logZZ_est = logZZ_est/10;
        Logprob = [ Logprob calculate_logprob(vishid,hidbiases,visbiases,logZZ_est,testbatchdata) ];
        Logprob2 = [ Logprob2 calculate_logprob(vishid,hidbiases,visbiases,logZZ_est,testbatchdata2) ];
    end
end