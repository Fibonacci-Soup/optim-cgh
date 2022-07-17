target = im2double(imread("Output_2D_iter\Target_field.png"));
% figure('DefaultAxesFontSize', 36);


MSE_list = [];
NMSE_list = [];
% PSNR_list= [];
for i = 1:50
    image = im2double(imread("Output_2D_iter\recon_i_" + string(i-1) + ".png"));
%     image = abs(fftshift(fft2(fftshift(exp(2*pi*j*im2double(imread("holo_i_" + string(i-1) + ".bmp")))))) / sum(size(target)));
    MSE_list(i) = immse(image, target);
    NMSE_list(i) = MSE_list(i) / sum(target(:).^2);
%     PSNR_list(i) = psnr(image, target);
end

% plot(MSE_list)
% xlabel("iterations")
% ylabel("MSE")

figure;
plot(NMSE_list)
xlabel("iterations")
ylabel("NMSE")

hold on
plot(GS_NMSE_list)

% figure;
% plot(PSNR_list)
% xlabel("iterations")
% ylabel("PSNR")


