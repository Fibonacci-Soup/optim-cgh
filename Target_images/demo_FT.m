ImageFileName = 'mandrill.png';
TargetImage = double(rgb2gray(imread(ImageFileName)));

Aperture = ifftshift(ifft2(ifftshift(TargetImage)));

imshow(abs(Aperture))
figure
imshow(angle(Aperture))