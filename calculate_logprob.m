function [logprob] = calculate_logprob(vishid,hidbiases,visbiases,logZ,batchdata)

numcases = size(batchdata,1);

pd = batchdata*visbiases' + sum(log(1+exp(ones(numcases,1)*hidbiases + batchdata*vishid)),2);
logprob = sum(pd)/numcases  - logZ;