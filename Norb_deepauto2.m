clear all;
close all;
clc;

maxepoch = 2^10;
beta = [0:1/1000:0.5 0.5:1/10000:0.9 0.9:1/10000:1.0];
numruns = 100;
numhid = 2000;

fid = fopen('smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r');
fread(fid,24,'uchar');
batchdata = zeros(100, 32*32*2, 243);
for i = 1 : 243
    for j = 1 : 100
        temp = edge(reshape(fread(fid,9216),96,96),'log');
        for k = 1 : 32
            for l = 1 : 32
                batchdata(j, (k-1)*32+l, i) = mean(mean(temp((k-1)*3+1:k*3,(l-1)*3+1:l*3)));
            end
        end
        temp = edge(reshape(fread(fid,9216),96,96),'log');
        for k = 1 : 32
            for l = 1 : 32
                batchdata(j, (k-1)*32+l+1024, i) = mean(mean(temp((k-1)*3+1:k*3,(l-1)*3+1:l*3)));
            end
        end
    end
end
clear temp;

% load smallnorb_32x32-training-dat;
% temp = batchdata;
% batchdata = zeros(100, 2048, 243);
% for i = 1 : 243
%     batchdata(:,:,i) = temp((i-1)*100+1:i*100, :);
% end
% clear temp;

fid = fopen('smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r');
fread(fid,24,'uchar');
testbatchdata = zeros(24300,32*32*2);
for i = 1 : 24300
    temp = edge(reshape(fread(fid,9216),96,96),'log');
    for j = 1 : 32
        for k = 1 : 32
            testbatchdata(i,(j-1)*32+k) = mean(mean(temp((j-1)*3+1:j*3,(k-1)*3+1:k*3)));
        end
    end
    temp = edge(reshape(fread(fid,9216),96,96),'log');
    for j = 1 : 32
        for k = 1 : 32
            testbatchdata(i,(j-1)*32+k+1024) = mean(mean(temp((j-1)*3+1:j*3,(k-1)*3+1:k*3)));
        end
    end
end
clear temp;

% load smallnorb_32x32-testing-dat;

testbatchdata2 = rand(243000,numhid) > rand(1000,numhid);

[numcases numdims numbatches]=size(batchdata);

numdims = numhid;
numhid = numpen;

vishid0 = 0.01*randn(numdims,numhid);
hidbiases0  = zeros(1,numhid);
visbiases0  = zeros(1,numdims);

rbm_rcd;
save norbrcd vishid hidbiases visbiases Error Logprob Logprob2;

figure(2);
plot(Error,'b-');
hold on;
figure(3);
plot(Logprob,'bo-');
hold on;
figure(4);
plot(Logprob2,'bo-');
hold on;

rbm_rpcd;
save norbrpcd vishid hidbiases visbiases Error Logprob Logprob2;

figure(2);
plot(Error,'r-');
hold on;
figure(3);
plot(Logprob,'rx-');
hold on;
figure(4);
plot(Logprob2,'rx-');
hold on;

rbm_rfpcd;
save norbrfpcd vishid hidbiases visbiases Error Logprob Logprob2;

figure(2);
plot(Error,'r-');
hold on;
figure(3);
plot(Logprob,'r+-');
hold on;
figure(4);
plot(Logprob2,'r+-');
hold on;
    
rbm_rpmcmc;
save norbrpmcmc vishid hidbiases visbiases Error Logprob Logprob2;

figure(2);
plot(Error,'k-');
hold on;
figure(3);
plot(Logprob,'k*-');
hold on;
figure(4);
plot(Logprob2,'k*-');
hold on;

rbm_rtmcmc;
save norbrtmcmc vishid hidbiases visbiases Error Logprob Logprob2;

figure(2);
plot(Error,'g-');
hold on;
figure(3);
plot(Logprob,'gs-');
hold on;
figure(4);
plot(Logprob2,'gs-');
hold on;

rbm_ramcmc;
save norbramcmc vishid hidbiases visbiases Error Logprob Logprob2;

figure(2);
plot(Error,'c-');
hold on;
figure(3);
plot(Logprob,'cd-');
hold on;
figure(4);
plot(Logprob2,'cd-');
hold on;