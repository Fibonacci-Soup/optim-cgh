img = imread('512_D.png');
size(img)
target_image = imread('Teapot_slices\Teapot_section_0.png');

% img = rgb2gray(img);

resized_photo = imresize(img, size(target_image));
imwrite(resized_photo, '1080p_D.png')