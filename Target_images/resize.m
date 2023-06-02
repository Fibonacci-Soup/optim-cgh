img = imread('mandrill2_square.png');
size(img)
target_image = imread('mandrill2_1280.png');

% img = rgb2gray(img);

resized_photo = imresize(img, size(target_image));
imwrite(resized_photo, 'mandrill2_stretched.png')