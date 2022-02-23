function generate_mod_LR_bic()

% put training data here
input_img = '/Users/hm_cai/PycharmProjects/CResMD_Testing/DIV2k_HR/0141.png';
save_LR_path = '/Users/hm_cai/PycharmProjects/CResMD_Testing/Noise_set';

for i=0:50
    % read image
    img = im2double(imread( input_img ));
    noise_label = i;
    noiseSigma = noise_label;

    noise = noiseSigma/255.*randn(size(img));
    im_noise = single(img + noise);
    im_noise = im2uint8(im_noise);

    kernel_label = 0;
    imwrite(im_noise, fullfile(save_LR_path, ['0141' '_' num2str(noise_label, '%02d') '.png']));
end








