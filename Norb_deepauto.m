clear all;
close all;
clc;

maxepoch = 2^10;
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

fid = fopen('smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r');
fread(fid,20,'uchar');
batchtarget = zeros(100,5,243);
for i = 1 : 243
    for j = 1 : 100
        batchtarget(j,:,i) = bitget(2^(4-fread(fid,1)),5:-1:1);
        fread(fid,3);
    end
end

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

fid = fopen('smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r');
fread(fid,20,'uchar');
testbatchtarget = zeros(24300,5);
for i = 1 : 24300
    testbatchtarget(i,:) = bitget(2^(4-fread(fid,1)),5:-1:1);
    fread(fid,3);
end

[numcases numdims numbatches]=size(batchdata);

numdims = numhid;
numhid = numpen;

vishid0 = 0.01*randn(numdims,numhid);
hidbiases0  = zeros(1,numhid);
visbiases0  = zeros(1,numdims);

numlabel=5;
labhid0 = 0.01*randn(numlabel,numhid);
labbiases0  = zeros(1,numlabel);

rbm_cd;
save norbcd vishid hidbiases visbiases labhid labbiases Error RBMClassification;

figure(2);
plot(Error,'b-');
hold on;
figure(3);
plot(RBMClassification,'bo-');
hold on;

rbm_pcd;
save norbpcd vishid hidbiases visbiases labhid labbiases Error RBMClassification;

figure(2);
plot(Error,'r-');
hold on;
figure(3);
plot(RBMClassification,'rx-');
hold on;

rbm_fpcd;
save norbfpcd1 vishid hidbiases visbiases labhid labbiases Error RBMClassification;

figure(2);
plot(Error,'r-');
hold on;
figure(3);
plot(RBMClassification,'r+-');
hold on;

rbm_pmcmc;
save norbpmcmc vishid hidbiases visbiases labhid labbiases Error RBMClassification;

figure(2);
plot(Error,'k-');
hold on;
figure(3);
plot(RBMClassification,'k*-');
hold on;

rbm_tmcmc;
save norbtmcmc vishid hidbiases visbiases labhid labbiases Error RBMClassification;

figure(2);
plot(Error,'g-');
hold on;
figure(3);
plot(RBMClassification,'gs-');
hold on;

rbm_amcmc;
save norbamcmc vishid hidbiases visbiases labhid labbiases Error RBMClassification;

figure(2);
plot(Error,'c-');
hold on;
figure(3);
plot(RBMClassification,'cd-');
hold on;