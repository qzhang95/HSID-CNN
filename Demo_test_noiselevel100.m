%% settings

clear;close all;

%% Model
def  = 'Model/HSID-CNN.prototxt';
model= 'Model/HSID-CNN_noiselevel100_iter_600000.caffemodel';

%% noise level
noiseSigma=100.0;

%% get GT HSI data
load('Data/GT_crop.mat');
im_label=temp;
[w,h, band] = size(im_label);
im_input=zeros(w,h, band);

%% add noise (same level for all bands)
for i=1:band
 
    im_input(:, :, i) = im_label(:, :, i) + noiseSigma/255.0*randn(size(im_label(:, :, i)));
end

%% HSI-denoising
caffe.reset_all();
caffe.set_mode_gpu();
net = caffe.Net(def, model, 'test');

k=12;
im_output=zeros(w, h, band);

for i=1 : 1 : k
    output1 = net.forward({im_input(:, :, 1:24),im_input(:, :, i)});
    output = output1{1,1};
    im_output(:, :, i) = im_input(:, :, i)+output;
end

for i=band-k+1 : 1 : band
    output1 = net.forward({im_input(:, :, 168:191),im_input(:, :, i)});
    output = output1{1,1};
    im_output(:, :, i) = im_input(:, :, i)+output;
end

for i=1+k : 1 : band-k
    output1 = net.forward({im_input(:, :, [i-k:i-1, i+1:i+k]),im_input(:, :, i)});
    output = output1{1,1};
    im_output(:, :, i) = im_input(:, :, i)+output;
end


%% PSNR & SSIM
PSNR=zeros(band, 1);
SSIM=zeros(band, 1);

for i=1:band
 
    [psnr_cur, ssim_cur, ~] = Cal_PSNRSSIM(im_output(:, :, i), im_label(:, :, i), 0, 0);
    PSNR(i,1)=psnr_cur;
    SSIM(i,1)=ssim_cur;
end

[SAM1, SAM2]=SAM(im_label, im_output);
disp(SAM1);

show_band=[57, 27, 17];

subplot(131), imshow(im_label(:, :, show_band));
title(['Original Band Number: ', num2str(show_band)])

subplot(132), imshow(im_input(:, :, show_band));
title(['Noise Level = ', num2str(floor(noiseSigma))])

subplot(133), imshow(im_output(:, :, show_band));
title(['MPSNR: ',num2str(mean(PSNR),'%2.4f'),'dB','    MSSIM: ',num2str(mean(SSIM),'%2.4f'),'    MSA: ',num2str(SAM1,'%2.4f')])

drawnow;

disp([mean(PSNR), mean(SSIM), SAM1]);


