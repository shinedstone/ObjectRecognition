%% Initializing symmetric weights and biases.
vishid = vishid0;
hidbiases  = hidbiases0;
visbiases  = visbiases0;
labhid = labhid0;
labbiases = labbiases0;

poshidprobs = zeros(numcases,numhid);
posprods    = zeros(numdims,numhid);
negprods    = zeros(numdims,numhid);
vishidinc  = zeros(numdims,numhid);
hidbiasinc = zeros(1,numhid);
visbiasinc = zeros(1,numdims);
labhidinc =  zeros(numlabel,numhid);
labbiasinc =  zeros(1,numlabel);
Error = zeros(1,maxepoch);
RBMClassification = [];

nchain = 100;
negdata = zeros(nchain,numdims);
neglabstates = zeros(nchain,numlabel);
for i = 1 : 500
    neghidprobs = 1./(1 + exp(-negdata*vishid-neglabstates*labhid-repmat(hidbiases,nchain,1)));
    neghidstates = neghidprobs > rand(nchain,numhid);
    negdata=1./(1 + exp(-neghidstates*vishid'-repmat(visbiases,nchain,1)));
    negdata = negdata > rand(nchain,numdims);
    neglabprobs = exp( neghidstates*labhid' + repmat(labbiases,nchain,1));
    neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlabel));
    xx = cumsum(neglabprobs,2);
    xx1 = rand(nchain,1);
    neglabstates = zeros(nchain,numlabel);
    for jj = 1 : nchain
        index = min(find(xx1(jj) <= xx(jj,:)));
        neglabstates(jj,index) = 1;
    end
end

for epoch = 1:maxepoch
    fprintf(1,'pcd - epoch %d\r',epoch);
    errsum = 0;
    epsilonw      = 0.01/(1+epoch/3000);
    epsilonvb     = 0.01/(1+epoch/3000);
    epsilonhb     = 0.01/(1+epoch/3000);
    for batch = 1:numbatches
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        target = batchtarget(:,:,batch);
        poshidprobs = 1./(1 + exp( - data*vishid - target*labhid - repmat(hidbiases,numcases,1)));
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        poslabprods = target'*poshidprobs;
        poslabact = sum(target);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        neghidprobs = 1./(1 + exp( - negdata*vishid - neglabstates*labhid - repmat(hidbiases,nchain,1) ));
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        neglabprods = neglabstates'*neghidprobs;
        neglabact = sum(neglabstates);
        
        neghidstates = neghidprobs > rand(nchain,numhid);
        
        neglabprobs = exp( neghidstates*labhid' + repmat(labbiases,nchain,1));
        neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlabel));
        xx = cumsum(neglabprobs,2);
        xx1 = rand(nchain,1);
        neglabstates = zeros(nchain,numlabel);
        for jj = 1 : nchain
            index = min(find(xx1(jj) <= xx(jj,:)));
            neglabstates(jj,index) = 1;
        end
        
        negdata = 1./(1 + exp( - neghidstates*vishid' - repmat(visbiases,nchain,1) ));
        negdata = negdata > rand(nchain,numdims);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = epsilonw*(posprods/numcases-negprods/(nchain));
        visbiasinc = epsilonvb*(posvisact/numcases-negvisact/(nchain));
        hidbiasinc = epsilonhb*(poshidact/numcases-neghidact/(nchain));
        labhidinc = epsilonw*(poslabprods/numcases-neglabprods/(nchain));
        labbiasinc = epsilonvb*(poslabact/numcases-neglabact/(nchain));
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        labhid = labhid + labhidinc;
        labbiases = labbiases + labbiasinc;
        
        errsum = errsum + sum((data-(1./(1+exp(-poshidstates*vishid'-repmat(visbiases,numcases,1)))>rand(numcases,numdims))).^2,2);
    end
    Error(epoch) = mean(errsum/numbatches);
    if rem(log2(epoch),1) == 0
        RBMClassification = [ RBMClassification calculate_classification_norb(vishid,hidbiases,visbiases,labhid,labbiases,testbatchdata,testbatchtarget) ];
        figure(4);
        plot(RBMClassification);
        drawnow;
    end
end