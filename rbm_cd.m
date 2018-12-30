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

for epoch = 1:maxepoch
    fprintf(1,'cd - epoch %d\r',epoch);
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
        neglabprobs = exp( poshidstates*labhid' + repmat(labbiases,numcases,1));
        neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlabel));
        xx = cumsum(neglabprobs,2);
        xx1 = rand(numcases,1);
        neglabstates = zeros(numcases,numlabel);
        for jj = 1 : numcases
            index = min(find(xx1(jj) <= xx(jj,:)));
            neglabstates(jj,index) = 1;
        end
        negdata = 1./(1 + exp( - poshidstates*vishid' - repmat(visbiases,numcases,1)));
        negdata = negdata > rand(numcases,numdims);
        neghidprobs = 1./(1 + exp( - negdata*vishid - neglabstates*labhid - repmat(hidbiases,numcases,1)));
        
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);
        neglabprods = neglabstates'*neghidprobs;
        neglabact = sum(neglabstates);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = epsilonw*(posprods-negprods)/numcases;
        visbiasinc = epsilonvb*(posvisact-negvisact)/numcases;
        hidbiasinc = epsilonhb*(poshidact-neghidact)/numcases;
        labhidinc = epsilonw*(poslabprods-neglabprods)/numcases;
        labbiasinc = epsilonvb*(poslabact-neglabact)/numcases;
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        labhid = labhid + labhidinc;
        labbiases = labbiases + labbiasinc;
        
        errsum = errsum + sum((data-negdata).^2,2);
    end
    Error(epoch) = mean(errsum/numbatches);
    if rem(log2(epoch),1) == 0
        RBMClassification = [ RBMClassification calculate_classification_norb(vishid,hidbiases,visbiases,labhid,labbiases,testbatchdata,testbatchtarget) ];
        figure(4);
        plot(RBMClassification);
        drawnow;
    end
end