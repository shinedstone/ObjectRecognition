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

temperature = 0.9:0.005:1;
nchain = size(temperature,2);
negdata = zeros(nchain,numdims);
neglabstates = zeros(nchain,numlabel);
for i = 1 : 500
    neghidprobs = 1./(1 + exp(-(negdata.*repmat(temperature',1,numdims))*vishid-(neglabstates.*repmat(temperature',1,numlabel))*labhid-temperature'*hidbiases));
    neghidstates = neghidprobs > rand(nchain,numhid);
    
    negdata=1./(1 + exp(-(neghidstates.*repmat(temperature',1,numhid))*vishid'-temperature'*visbiases));
    negdata = negdata > rand(nchain,numdims);
    
    neglabprobs = exp( (neghidstates.*repmat(temperature',1,numhid))*labhid' + temperature'*labbiases);
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
    fprintf(1,'tmcmc - epoch %d\r',epoch);
    errsum = 0;
    epsilonw      = 0.01/(1+epoch/3000);
    epsilonvb     = 0.01/(1+epoch/3000);
    epsilonhb     = 0.01/(1+epoch/3000);
    for batch = 1:numbatches
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        target = batchtarget(:,:,batch);
        poshidprobs = 1./(1 + exp( - data*vishid - target*labhid - repmat(hidbiases,numcases,1)));
        posprods=data'*poshidprobs;
        poshidact=sum(poshidprobs);
        posvisact=sum(data);
        poslabprods = target'*poshidprobs;
        poslabact = sum(target);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        %%%%%%%%% START NEGATIVE PHASE   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i = 1 : nchain-1
            E11 = -(negdata(i,:)*(visbiases*temperature(i))'+neglabstates(i,:)*(labbiases*temperature(i))'+sum(log(1+exp(hidbiases*temperature(i)+negdata(i,:)*(vishid*temperature(i))+neglabstates(i,:)*(labhid*temperature(i)))),2));
            E11s = -(negdata(i,:)*(visbiases*temperature(i+1))'+neglabstates(i,:)*(labbiases*temperature(i+1))'+sum(log(1+exp(hidbiases*temperature(i+1)+negdata(i,:)*(vishid*temperature(i+1))+neglabstates(i,:)*(labhid*temperature(i+1)))),2));
            E22 = -(negdata(i+1,:)*(visbiases*temperature(i+1))'+neglabstates(i+1,:)*(labbiases*temperature(i+1))'+sum(log(1+exp(hidbiases*temperature(i+1)+negdata(i+1,:)*(vishid*temperature(i+1))+neglabstates(i+1,:)*(labhid*temperature(i+1)))),2));
            E22s = -(negdata(i+1,:)*(visbiases*temperature(i))'+neglabstates(i+1,:)*(labbiases*temperature(i))'+sum(log(1+exp(hidbiases*temperature(i)+negdata(i+1,:)*(vishid*temperature(i))+neglabstates(i+1,:)*(labhid*temperature(i)))),2));
            
            swap_prob = min( 1 , exp(E11 - E11s + E22 - E22s) );
            
            swapping_particles = binornd(1, swap_prob);
            staying_particles = 1 - swapping_particles;
            
            swp_p_visible = repmat(swapping_particles, [1 numdims]);
            swp_p_label = repmat(swapping_particles, [1 numlabel]);
            sty_p_visible = repmat(staying_particles, [1 numdims]);
            sty_p_label = repmat(staying_particles, [1 numlabel]);
            
            v1t1 = negdata(i,:).*swp_p_visible;
            l1t1 = neglabstates(i,:).*swp_p_label;
            v1t2 = negdata(i+1,:).*swp_p_visible;
            l1t2 = neglabstates(i+1,:).*swp_p_label;
            
            negdata(i,:) = negdata(i,:).*sty_p_visible;
            neglabstates(i,:) = neglabstates(i,:).*sty_p_label;
            negdata(i+1,:) = negdata(i+1,:).*sty_p_visible;
            neglabstates(i+1,:) = neglabstates(i+1,:).*sty_p_label;
            
            negdata(i,:) = negdata(i,:)+v1t2;
            neglabstates(i,:) = neglabstates(i,:)+l1t2;
            negdata(i+1,:) = negdata(i+1,:)+v1t1;
            neglabstates(i+1,:) = neglabstates(i+1,:)+l1t1;
        end
        
        neghidprobs = 1./(1+exp(-negdata(end,:)*vishid-neglabstates(end,:)*labhid-hidbiases));
        
        negprods=negdata(end,:)'*neghidprobs;
        neghidact=neghidprobs;
        negvisact=negdata(end,:);
        neglabprods = neglabstates(end,:)'*neghidprobs;
        neglabact = neglabstates(end,:);
        
        neghidprobs = 1./(1 + exp(-(negdata.*repmat(temperature',1,numdims))*vishid-(neglabstates.*repmat(temperature',1,numlabel))*labhid-temperature'*hidbiases));
        neghidstates = neghidprobs > rand(nchain,numhid);
        
        negdata = 1./(1 + exp(-(neghidstates.*repmat(temperature',1,numhid))*vishid'-temperature'*visbiases));
        negdata = negdata > rand(nchain,numdims);
        
        neglabprobs = exp( (neghidstates.*repmat(temperature',1,numhid))*labhid' + temperature'*labbiases);
        neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlabel));
        xx = cumsum(neglabprobs,2);
        xx1 = rand(nchain,1);
        neglabstates = zeros(nchain,numlabel);
        for jj = 1 : nchain
            index = min(find(xx1(jj) <= xx(jj,:)));
            neglabstates(jj,index) = 1;
        end
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        vishidinc = epsilonw*(posprods/numcases-negprods);
        visbiasinc = epsilonvb*(posvisact/numcases-negvisact);
        hidbiasinc = epsilonhb*(poshidact/numcases-neghidact);
        labhidinc = epsilonw*(poslabprods/numcases-neglabprods);
        labbiasinc = epsilonvb*(poslabact/numcases-neglabact);
        
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