function generate_mod_LR_bic()

% put training data here
input_img = '/Users/hm_cai/PycharmProjects/CResMD_Testing/DIV2k_HR/0141.png';
save_LR_path = '/Users/hm_cai/PycharmProjects/CResMD_Testing/Blur_set';

for i=0:25
    % read image
    img = im2double(imread( input_img ));
    kernel_label = i;
    kernelwidth = kernel_label/10.;

    if kernel_label > 0
        kernel = single(fspecial('gaussian', 21, kernelwidth));
        blurry_img = imfilter(img, double(kernel), 'replicate');
    else
        blurry_img = img;
    end
    imwrite(blurry_img, fullfile(save_LR_path, ['0141' '_' num2str(kernel_label, '%02d') '.png']));
end


