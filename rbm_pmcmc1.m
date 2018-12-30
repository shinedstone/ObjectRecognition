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
nFalse = [];

temperature = 0.901:0.001:1;
ntemp = 100;
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

tempdata = negdata;
templabstates = neglabstates;

for epoch = 1:maxepoch
    fprintf(1,'pmcmc - epoch %d\r',epoch);
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
        temphidprobs = 1./(1 + exp( -tempdata*vishid -templabstates*labhid -repmat(hidbiases,nchain,1)));
        negprods  = tempdata'*temphidprobs;
        neghidact = sum(temphidprobs);
        negvisact = sum(tempdata);
        neglabprods = templabstates'*temphidprobs;
        neglabact = sum(templabstates);

        neghidprobs = 1./(1 + exp( -(negdata.*repmat(temperature', 1, numdims))*vishid -(neglabstates.*repmat(temperature', 1, numlabel))*labhid -repmat(hidbiases,nchain,1)));
        neghidstates = neghidprobs > rand(nchain,numhid);
        
        neglabprobs = exp( (neghidstates.*repmat(temperature', 1, numhid))*labhid' + repmat(labbiases,nchain,1));
        neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlabel));
        xx = cumsum(neglabprobs,2);
        xx1 = rand(nchain,1);
        neglabstates = zeros(nchain,numlabel);
        for jj = 1 : nchain
            index = min(find(xx1(jj) <= xx(jj,:)));
            neglabstates(jj,index) = 1;
        end
        
        negdata = 1./(1 + exp( -(neghidstates.*repmat(temperature', 1, numhid))*vishid' -repmat(visbiases,nchain,1)));
        negdata = negdata > rand(nchain,numdims);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = epsilonw*(posprods/numcases-negprods/nchain);
        visbiasinc = epsilonvb*(posvisact/numcases-negvisact/nchain);
        hidbiasinc = epsilonhb*(poshidact/numcases-neghidact/nchain);
        labhidinc = epsilonw*(poslabprods/numcases-neglabprods/nchain);
        labbiasinc = epsilonvb*(poslabact/numcases-neglabact/nchain);
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        labhid = labhid + labhidinc;
        labbiases = labbiases + labbiasinc;
        
        x = negdata*visbiasinc'+neglabstates*labbiasinc'+sum(log(1+exp(repmat(hidbiases,nchain,1)+negdata*vishid+neglabstates*labhid)),2)-sum(log(1+exp(repmat(hidbiases-hidbiasinc,nchain,1)+negdata*(vishid-vishidinc)+neglabstates*(labhid-labhidinc))),2);
        x = exp(x - max(x));
        x = x/sum(x);
        xx = cumsum(x);
        xx1 = rand(nchain,1);
        tempdata = zeros(nchain, numdims);
        templabstates = zeros(nchain, numlabel);
        for jj = 1 : nchain
            index = min(find(xx1(jj) <= xx));
            tempdata(jj,:) = negdata(index,:);
            templabstates(jj,:) = neglabstates(index,:);
        end

        errsum = errsum + sum((data-(1./(1+exp(-poshidstates*vishid'-repmat(visbiases,numcases,1)))>rand(numcases,numdims))).^2,2);
    end
    Error(epoch) = mean(errsum/numbatches);
    
    if epoch > maxepoch*0.1
        [ falsedata, falsetarget, nBatches ] = make_samples(vishid,hidbiases,visbiases,labhid,labbiases,batchdata,batchtarget);
        nFalse = [ nFalse nBatches*numcases ];
        figure(5);
        plot(nFalse);
        drawnow;
        
        for batch = 1 : nBatches
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = falsedata(:,:,batch);
            target = falsetarget(:,:,batch);
            poshidprobs = 1./(1 + exp( - data*vishid - target*labhid - repmat(hidbiases,numcases,1)));
            posprods    = data' * poshidprobs;
            poshidact   = sum(poshidprobs);
            posvisact = sum(data);
            poslabprods = target'*poshidprobs;
            poslabact = sum(target);
            
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs > rand(numcases,numhid);
            
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            temphidprobs = 1./(1 + exp( -tempdata*vishid -templabstates*labhid -repmat(hidbiases,nchain,1)));
            negprods  = tempdata'*temphidprobs;
            neghidact = sum(temphidprobs);
            negvisact = sum(tempdata);
            neglabprods = templabstates'*temphidprobs;
            neglabact = sum(templabstates);
            
            neghidprobs = 1./(1 + exp( -(negdata.*repmat(temperature', 1, numdims))*vishid -(neglabstates.*repmat(temperature', 1, numlabel))*labhid -temperature'*hidbiases));
            neghidstates = neghidprobs > rand(nchain,numhid);
            
            neglabprobs = exp( (neghidstates.*repmat(temperature', 1, numhid))*labhid' + temperature'*labbiases);
            neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlabel));
            xx = cumsum(neglabprobs,2);
            xx1 = rand(nchain,1);
            neglabstates = zeros(nchain,numlabel);
            for jj = 1 : nchain
                index = min(find(xx1(jj) <= xx(jj,:)));
                neglabstates(jj,index) = 1;
            end
            
            negdata = 1./(1 + exp( -(neghidstates.*repmat(temperature', 1, numhid))*vishid' -temperature'*visbiases));
            negdata = negdata > rand(nchain,numdims);
            
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = 3*epsilonw*(posprods/numcases-negprods/nchain);
            visbiasinc = 3*epsilonvb*(posvisact/numcases-negvisact/nchain);
            hidbiasinc = 3*epsilonhb*(poshidact/numcases-neghidact/nchain);
            labhidinc = 3*epsilonw*(poslabprods/numcases-neglabprods/nchain);
            labbiasinc = 3*epsilonvb*(poslabact/numcases-neglabact/nchain);
            
            vishid = vishid - vishidinc;
            visbiases = visbiases - visbiasinc;
            hidbiases = hidbiases - hidbiasinc;
            labhid = labhid - labhidinc;
            labbiases = labbiases - labbiasinc;
            
            x = negdata*visbiasinc'+neglabstates*labbiasinc'+sum(log(1+exp(repmat(hidbiases,nchain,1)+negdata*vishid+neglabstates*labhid)),2)-sum(log(1+exp(repmat(hidbiases-hidbiasinc,nchain,1)+negdata*(vishid-vishidinc)+neglabstates*(labhid-labhidinc))),2);
            x = exp(x - max(x));
            x = x/sum(x);
            xx = cumsum(x);
            xx1 = rand(nchain,1);
            tempdata = zeros(nchain, numdims);
            templabstates = zeros(nchain, numlabel);
            for jj = 1 : nchain
                index = min(find(xx1(jj) <= xx));
                tempdata(jj,:) = negdata(index,:);
                templabstates(jj,:) = neglabstates(index,:);
            end
        end
    end
    
    if rem(log2(epoch),1) == 0
        RBMClassification = [ RBMClassification calculate_classification_norb(vishid,hidbiases,visbiases,labhid,labbiases,testbatchdata,testbatchtarget) ];
        figure(4);
        plot(RBMClassification);
        drawnow;
    end
end