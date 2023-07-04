%%% Written for CMMPE by Daoming Dong and Youchao Wang, 
%%% Copyright 2018-2019

close all;
clear;
clc;

% for FileNameCount = 1: 6
% ImageFileName = sprintf('output1202%d.bmp', FileNameCount);
ImageFileName = 'holography_ambigram_1024.png';
TargetImage = double(imread(ImageFileName));%/255;
% TargetImage = double(rgb2gray(imread(ImageFileName)));%/255;

%%% The following commented-out code might be useful
    % Horizontal flip
%     TargetImage = flipdim(TargetImage, 2);
    
    % Image normalization
    % TargetImage = abs(ifft2(real(fft2(TargetImage))));
    % TargetImage = TargetImage/max(TargetImage(:));
%%%

% A reference ray in phase is defined with energy of 1

Ref = exp(1i*ones(size(TargetImage))*2*pi*rand(1)); %change rand(1) to%0.0720); % previously rand(1)
Ref = Ref/sqrt(sum(abs(Ref(:)).^2)); % Euclidean norm
    
% Number of sub-frames to average:
N = 8;
recont = 0;
% A = sqrt(TargetImage);
A = TargetImage;

% Assumming we want to output three channels in the Freeman projector

% Major loop for hologram generation
for TotalTotalIndex = 1:3
    TotalPhaseRGB = 0;
    for j = 1:N
    Diffuser = exp(1i*2*pi* rand(size(TargetImage)));

    Efield = Ref.*A;
    Efield = Efield .*Diffuser;
    Enr = sum(abs(Efield(:)).^2)/N; 

    Farfield = (ifft2(ifftshift(Efield)));
    % Energy conservation
    Farfield = Farfield/sqrt(sum(abs(Farfield(:)).^2)/Enr);
    % Conservation actually not needed for pure demonstration purpose 

    % Returns the angle of each complex element
    PhaseOfHologram = angle(Farfield);

    % Binary phase quantization
    PhaseOfHologram = double(PhaseOfHologram > 0);

    TotalPhaseRGB = TotalPhaseRGB + PhaseOfHologram .* 2^(j-1);
    end
    TotalTotal(:,:,TotalTotalIndex) = uint8(TotalPhaseRGB);
end

%  imshow(TotalPhaseRGB, []);
 imshow(TotalTotal);
 imwrite(TotalTotal, "TestHologram_" + ImageFileName,'bmp');
% end