function Single_Degradation22()
clear all

% put training data here

input_path = '/media/sdf1/HMCai/CResMD-GAN_Training/DIV2K_Train_HR_sub';
save_LR_path = '/media/sdf1/HMCai/CResMD-GAN_Training/DIV2K_Train_HR_sub_Specific_Degradation/DIV2K_Train_HR_sub_k0_noise50';

file_type = '.png';



% Gaussian_Noise / Blur_Noise / JPEG / 'Noise_Blur'
noise_mode = 'Noise_Blur';


if exist('save_mod_path', 'var')
    if exist(save_mod_path, 'dir')
        disp(['It will cover ', save_mod_path]);
    else
        mkdir(save_mod_path);
    end
end
if exist('save_LR_path', 'var')
    if exist(save_LR_path, 'dir')
        disp(['It will cover ', save_LR_path]);
    else
        mkdir(save_LR_path);
    end
end

randn('seed', 0);

% parpool('local');
filepaths = dir(fullfile(input_path,'*.*'));
parfor i = 1 : length(filepaths)
    

    % kernel [0:50]
    kernel_label = 0;
    % noise  [0:50]
    noise_label = 50;
    % Jpeg [10:40]
    jpeg_quality = 0;

    
    
    
    
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        str_rlt = sprintf('%d\t%s.\n', i, imname);
        fprintf(str_rlt);

        % read image
        img = im2double(imread(fullfile(input_path, [imname, ext])));
        
        if strcmp(noise_mode,'Gaussian_Noise')
            % Create AGWN noisy img
            noiseSigma = noise_label;
            noise = noiseSigma/255.*randn(size(img));
            im_noise = single(img + noise);
            result_img = im2uint8(im_noise);
            kernel_label = 0;
            jpeg_quality = 0;
            imwrite(result_img, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') num2str(jpeg_quality, '%02d') '.png']));
        
        elseif strcmp(noise_mode, 'Blur_Noise')
            % Create Blur img 
            kernelwidth = kernel_label/10.;
            kernel = single(fspecial('gaussian', 21, kernelwidth));
            result_img = imfilter(img, double(kernel), 'replicate');
            noise_label = 0;
            jpeg_quality = 0;
            imwrite(result_img, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') num2str(jpeg_quality, '%02d') '.png']));
            
        elseif strcmp(noise_mode,'JPEG')
            noise_label = 0;
            kernel_label = 0;
            imwrite(img, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') num2str(jpeg_quality, '%02d') '.jpg']), 'jpg', 'Quality', jpeg_quality);
        
        elseif strcmp(noise_mode,'Noise_Blur')
            %kernelwidth = kernel_label/10.;
            %kernel = single(fspecial('gaussian', 21, kernelwidth));
            %blur_img = imfilter(img, double(kernel), 'replicate');


            noiseSigma = noise_label;
            %noise = noiseSigma/255.*randn(size(blur_img));
            %im_noise = single(blur_img + noise);
            noise = noiseSigma/255.*randn(size(img));
            im_noise = single(img + noise);
            result_img = im2uint8(im_noise);
            imwrite(result_img, fullfile(save_LR_path, [imname '_' '00' num2str(noise_label, '%02d') '.png']));




            %imwrite(result_img, fullfile(save_LR_path, [imname '_' '00' num2str(noise_label, '%02d') '.png']));

            %imwrite(result_img, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') '.png']));
            %if kernel_label ~= 0 and noise_label ~= 0
            %    imwrite(result_img, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') '.png']));
            %elseif kernel_label == 0 and noise_label ~= 0
            %    imwrite(result_img, fullfile(save_LR_path, [imname '_' '00' num2str(noise_label, '%02d') '.png']));
            %elseif kernel_label ~= 0 and noise_label == 0
            %    imwrite(result_img, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') '00' '.png']));
            %end
        
        end
    end


    
    
end    
end
