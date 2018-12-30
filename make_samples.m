function [ falsedata, falsetarget, nBatches ] = make_samples(vishid,hidbiases,visbiases,labhid,labbiases,batchdata,batchtarget)
[ numcases numdims numbatches ] = size(batchdata);
numlabel = size(batchtarget, 2);
tempdata = [];
temptarget = [];
for i = 1 : numbatches
    table = zeros(numcases, numlabel);
    for j = 1 : numlabel
        temp = repmat(bitget(2^(numlabel-j), numlabel:-1:1), numcases, 1);
        table(:, j) = (temp*labbiases')+sum(log(1+exp(repmat(hidbiases, numcases, 1)+batchdata(:, :, i)*vishid+temp*labhid)),2);
    end
    flag = table > repmat(sum(batchtarget(:, :, i).*table, 2), 1, numlabel);
    for j = 1 : numcases
        for k = 1 : numlabel
            if flag(j, k)
                tempdata = [ tempdata ; batchdata(j, :, i) ];
                temptarget = [ temptarget ; bitget(2^(numlabel-k), numlabel:-1:1) ];
            end
        end
    end
end
nSamps = size(tempdata, 1);
nBatches = floor(nSamps/100);
falsedata = zeros(numcases, numdims, nBatches);
falsetarget = zeros(numcases, numlabel, nBatches);
index = randperm(nSamps);
for i = 1 : nBatches
    falsedata(:, :, i) = tempdata(index(numcases*(i-1)+1:numcases*i), :);
    falsetarget(:, :, i) = temptarget(index(numcases*(i-1)+1:numcases*i), :);
end