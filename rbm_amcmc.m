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

nchain = 50;
slownegdata = zeros(nchain,numdims);
slowneglabstates = zeros(nchain,numlabel);
fastnegdata = zeros(nchain,numdims);
fastneglabstates = zeros(nchain,numlabel);
for i = 1 : 500
    slowneghidprobs = 1./(1 + exp(-slownegdata*vishid-slowneglabstates*labhid-repmat(hidbiases,nchain,1)));
    slowneghidstates = slowneghidprobs > rand(nchain,numhid);
    slownegdata=1./(1 + exp(-slowneghidstates*vishid'-repmat(visbiases,nchain,1)));
    slownegdata = slownegdata > rand(nchain,numdims);
    slowneglabprobs = exp( slowneghidstates*labhid' + repmat(labbiases,nchain,1));
    slowneglabprobs = slowneglabprobs./(sum(slowneglabprobs,2)*ones(1,numlabel));
    xx = cumsum(slowneglabprobs,2);
    xx1 = rand(nchain,1);
    slowneglabstates = zeros(nchain,numlabel);
    for jj = 1 : nchain
        index = min(find(xx1(jj) <= xx(jj,:)));
        slowneglabstates(jj,index) = 1;
    end
    fastneghidprobs = 1./(1 + exp(-fastnegdata*vishid-fastneglabstates*labhid-repmat(hidbiases,nchain,1)));
    fastneghidstates = fastneghidprobs > rand(nchain,numhid);
    fastnegdata=1./(1 + exp(-fastneghidstates*vishid'-repmat(visbiases,nchain,1)));
    fastnegdata = fastnegdata > rand(nchain,numdims);
    fastneglabprobs = exp( fastneghidstates*labhid' + repmat(labbiases,nchain,1));
    fastneglabprobs = fastneglabprobs./(sum(fastneglabprobs,2)*ones(1,numlabel));
    xx = cumsum(fastneglabprobs,2);
    xx1 = rand(nchain,1);
    fastneglabstates = zeros(nchain,numlabel);
    for jj = 1 : nchain
        index = min(find(xx1(jj) <= xx(jj,:)));
        fastneglabstates(jj,index) = 1;
    end
end

temperature = 0.9:0.005:1;
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
        target = batchtarget(:,:,batch);
        poshidprobs = 1./(1 + exp( - data*vishid - target*labhid - repmat(hidbiases,numcases,1)));
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        poslabprods = target'*poshidprobs;
        poslabact = sum(target);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fastneghidprobs=1./(1+exp(-(fastnegdata*vishid+fastneglabstates*labhid+repmat(hidbiases,nchain,1)).*repmat(currenttemperature,1,numhid)));
        fastneghidstates = fastneghidprobs > rand(nchain,numhid);
        
        fastnegdata=1./(1 + exp((-fastneghidstates*vishid'-repmat(visbiases,nchain,1)).*repmat(currenttemperature,1,numdims)));
        fastnegdata = fastnegdata > rand(nchain,numdims);
        
        fastneglabprobs = exp( (fastneghidstates*labhid' + repmat(labbiases,nchain,1)).*repmat(currenttemperature,1,numlabel));
        fastneglabprobs = fastneglabprobs./(sum(fastneglabprobs,2)*ones(1,numlabel));
        xx = cumsum(fastneglabprobs,2);
        xx1 = rand(nchain,1);
        fastneglabstates = zeros(nchain,numlabel);
        for jj = 1 : nchain
            index = min(find(xx1(jj) <= xx(jj,:)));
            fastneglabstates(jj,index) = 1;
        end
        
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
        
        E11 = -((fastnegdata.*repmat(currenttemperature,1,numdims))*visbiases'+(fastneglabstates.*repmat(currenttemperature,1,numlabel))*labbiases'...
            +sum(log(1+exp(currenttemperature*hidbiases+(fastnegdata.*repmat(currenttemperature,1,numdims))*vishid+(fastneglabstates.*repmat(currenttemperature,1,numlabel))*labhid)),2));
        E22 = -((fastnegdata.*repmat(previoustemperature,1,numdims))*visbiases'+(fastneglabstates.*repmat(previoustemperature,1,numlabel))*labbiases'...
            +sum(log(1+exp(previoustemperature*hidbiases+(fastnegdata.*repmat(previoustemperature,1,numdims))*vishid+(fastneglabstates.*repmat(previoustemperature,1,numlabel))*labhid)),2));
        
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
        
        slowneghidprobs = 1./(1 + exp(-slownegdata*vishid-slowneglabstates*labhid-repmat(hidbiases,nchain,1)));
        slowneghidstates = slowneghidprobs > rand(nchain,numhid);
        
        negprods=slownegdata'*slowneghidprobs;
        neghidact=sum(slowneghidprobs);
        negvisact=sum(slownegdata);
        neglabprods = slowneglabstates'*slowneghidprobs;
        neglabact = sum(slowneglabstates);
        
        slownegdata = 1./(1+exp(-slowneghidstates*vishid'-repmat(visbiases,nchain,1)));
        slownegdata = slownegdata > rand(nchain,numdims);

        slowneglabprobs = exp( slowneghidstates*labhid' + repmat(labbiases,nchain,1));
        slowneglabprobs = slowneglabprobs./(sum(slowneglabprobs,2)*ones(1,numlabel));
        xx = cumsum(slowneglabprobs,2);
        xx1 = rand(nchain,1);
        slowneglabstates = zeros(nchain,numlabel);
        for jj = 1 : nchain
            index = min(find(xx1(jj) <= xx(jj,:)));
            slowneglabstates(jj,index) = 1;
        end
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