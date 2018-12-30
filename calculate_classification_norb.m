function [result] = calculate_classification_norb(vishid,hidbiases,visbiases,labhid,labbiases,testdata,testtarget)
nSamps = size(testdata,1);
energy = zeros(nSamps,1);
numlabel = size(testtarget, 2);
rbm_class = zeros(nSamps,numlabel);
for i = 1 : numlabel
    previous_energy = energy;
    previous_class = rbm_class;
    rbm_class = repmat(bitget(2^(numlabel-i),numlabel:-1:1),nSamps,1);
    energy = (testdata*visbiases'+rbm_class*labbiases')+sum(log(1+exp(repmat(hidbiases,nSamps,1)+testdata*vishid+rbm_class*labhid)),2);
    swap_flag = energy > previous_energy;
    stay_flag = 1 - swap_flag;
    energy = swap_flag.*energy + stay_flag.*previous_energy;
    rbm_class = repmat(swap_flag,1,numlabel).*rbm_class + repmat(stay_flag,1,numlabel).*previous_class;
end
result = sum(sum(abs(testtarget - rbm_class),2)==0,1)/nSamps*100;