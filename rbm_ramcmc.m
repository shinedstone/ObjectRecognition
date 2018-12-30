%% Initializing symmetric weights and biases.
vishid = vishid0;
hidbiases  = hidbiases0;
visbiases  = visbiases0;

poshidprobs = zeros(numcases,numhid);
neghidprobs = zeros(nchain,numhid);
posprods    = zeros(numdims,numhid);
negprods    = zeros(numdims,numhid);
vishidinc  = zeros(numdims,numhid);
hidbiasinc = zeros(1,numhid);
visbiasinc = zeros(1,numdims);
batchposhidprobs = zeros(numcases,numhid,numbatches);
Error = zeros(1,maxepoch);
Logprob = [];
Logprob2 = [];

nchain = 50;
slownegdata = zeros(nchain,numdims);
fastnegdata = zeros(nchain,numdims);
for i = 1 : 500
    slowneghidprobs = 1./(1 + exp(-slownegdata*vishid-repmat(hidbiases,nchain,1)));
    slowneghidstates = slowneghidprobs > rand(nchain,numhid);
    slownegdata=1./(1 + exp(-slowneghidstates*vishid'-repmat(visbiases,nchain,1)));
    slownegdata = slownegdata > rand(nchain,numdims);
    
    fastneghidprobs = 1./(1 + exp(-fastnegdata*vishid-repmat(hidbiases,nchain,1)));
    fastneghidstates = fastneghidprobs > rand(nchain,numhid);
    fastnegdata=1./(1 + exp(-fastneghidstates*vishid'-repmat(visbiases,nchain,1)));
    fastnegdata = fastnegdata > rand(nchain,numdims);
end

temperature = 0.9:0.0033:1;
ntemp = size(temperature,2);
adapweight = ones(ntemp,1);
currentadapweight = zeros(ntemp,1);
currenttemp = ntemp*ones(nchain,1);
currenttemperature = zeros(nchain,1);
for i = 1 : nchain
    currenttemperature(i) = temperature(currenttemp(i));
    currentadapweight(i) = adapweight(currenttemp(i));
end

for epoch = 1:maxepoch
    fprintf(1,'amcmc - epoch %d\r',epoch);
    errsum = 0;
    epsilonw      = 0.01/(1+epoch/3000);
    epsilonvb     = 0.01/(1+epoch/3000);
    epsilonhb     = 0.01/(1+epoch/3000);
    for batch = 1:numbatches
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
        batchposhidprobs(:,:,batch) = poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fastneghidprobs=1./(1+exp(-(fastnegdata*vishid+repmat(hidbiases,nchain,1)).*repmat(currenttemperature,1,numhid)));
        fastneghidstates = fastneghidprobs > rand(nchain,numhid);
        
        fastnegdata=1./(1 + exp((-fastneghidstates*vishid'-repmat(visbiases,nchain,1)).*repmat(currenttemperature,1,numdims)));
        fastnegdata = fastnegdata > rand(nchain,numdims);
        
        previoustemp = currenttemp;
        previoustemperature = currenttemperature;
        previousadapweight = currentadapweight;
        
        for i = 1 : nchain
            if currenttemp(i) == ntemp
                currenttemp(i) = currenttemp(i) - 1;
            elseif currenttemp(i) == 1
                currenttemp(i) = currenttemp(i) + 1;
            else
                flag = randi(2);
                if flag == 2
                    currenttemp(i) = currenttemp(i) + 1;
                else
                    currenttemp(i) = currenttemp(i) - 1;
                end
            end
        end
        for i = 1 : nchain
            currenttemperature(i) = temperature(currenttemp(i));
            currentadapweight(i) = adapweight(currenttemp(i));
        end
        
        E11 = -((fastnegdata.*repmat(currenttemperature,1,numdims))*visbiases'...
            +sum(log(1+exp((currenttemperature*hidbiases+(fastnegdata.*repmat(currenttemperature,1,numdims))*vishid))),2));
        E22 = -((fastnegdata.*repmat(previoustemperature,1,numdims))*visbiases'...
            +sum(log(1+exp((previoustemperature*hidbiases+(fastnegdata.*repmat(previoustemperature,1,numdims))*vishid))),2));
        
        swap_prob = min( ones(nchain,1) , exp( -E11 +E22 ).*previousadapweight./currentadapweight );
        swapping_particles = binornd(1, swap_prob);
        staying_particles = 1 - swapping_particles;
        
        for i = 1 : nchain
            if staying_particles(i) == 1
                adapweight(previoustemp(i)) = adapweight(previoustemp(i))*0.999;
            end
        end
        
        currenttemp = previoustemp.*staying_particles + currenttemp.*swapping_particles;
        for i = 1 : nchain
            currenttemperature(i) = temperature(currenttemp(i));
            currentadapweight(i) = adapweight(currenttemp(i));
        end
        
        if numel(currenttemp(currenttemp==ntemp)) ~= 0
            for i = 1 : nchain
                if currenttemp(i) == ntemp
                    temp = fastnegdata(i,:);
                    fastnegdata(i,:) = slownegdata(i,:);
                    slownegdata(i,:) = temp;
                end
            end
        end
        
        slowneghidprobs=1./(1+exp(-slownegdata*vishid-repmat(hidbiases,nchain,1)));
        slowneghidstates = slowneghidprobs > rand(nchain,numhid);
        
        negprods=slownegdata'*slowneghidprobs;
        neghidact=sum(slowneghidprobs);
        negvisact=sum(slownegdata);
        
        slownegdata = 1./(1+exp(-slowneghidstates*vishid'-repmat(visbiases,nchain,1)));
        slownegdata = slownegdata > rand(nchain,numdims);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = epsilonw*(posprods/numcases-negprods/nchain);
        visbiasinc = epsilonvb*(posvisact/numcases-negvisact/nchain);
        hidbiasinc = epsilonhb*(poshidact/numcases-neghidact/nchain);
        
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