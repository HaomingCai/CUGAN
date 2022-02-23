function generate_mod_LR_bic()

% put test data here
% input_path = '/Users/hm_cai/CResMD_GAN_Download/Ori/Set14';
% save_LR_root = '/Users/hm_cai/CResMD_GAN_Download/Ori/Set14_Noised/ForCResMD';
% input_path = '/Users/hm_cai/CResMD_GAN_Download/Ori/Set14';
% save_LR_root = '/Users/hm_cai/CResMD_GAN_Download/Ori/Set14_Noised/ForCResMD/LowDegradation';
% input_path = '/Users/hm_cai/CResMD_GAN_Download/Ori/CBSD68_HR';
% save_LR_root = '/Users/hm_cai/CResMD_GAN_Download/Ori/Set14_Noised/ForCResMD/LowDegradation';
% input_path = '/Users/hm_cai/CResMD_GAN_Download/Ori/live1';
% save_LR_root = '/Users/hm_cai/CResMD_GAN_Download/Ori/live1_Noised/ForUpperBound';

input_path = '/Users/hm_cai/Paper_Project/CVPR2021_CUGAN/CResMD_GAN_Download/Ori/CBSD68';
save_LR_root = '/Users/hm_cai/Paper_Project/CVPR2021_CUGAN/CResMD_GAN_Download/Ori/CBSD68_Noised';
file_type = '.bmp';

% % kernelFor
% kernel_list = [0, 1, 2, 3, 4];
% kernel_label_list = kernel_list*10;
% 
% % noise
% noise_list = [0, 10, 20, 30, 40, 50];
% noise_label_list = noise_list;


% kernel
% kernel_list = [0.5, 1, 1.2, 1.5, 1.8, 2];
kernel_list = [0];
kernel_label_list = kernel_list*10;

% noise
noise_list = [25];
noise_label_list = noise_list;


for k_ind = 1:length(kernel_list)
    kernelwidth = kernel_list(k_ind);
    kernel_label = kernel_label_list(k_ind);
    if kernel_label > 0
        kernel = single(fspecial('gaussian', 21, kernelwidth));
    end 

    for s_ind = 1:length(noise_list)
        noiseSigma = noise_list(s_ind);
        noise_label = noise_label_list(s_ind);

        save_LR_path = fullfile(save_LR_root, ['blur' num2str(kernel_label) '_noise' num2str(noise_label)]);

        if exist('save_LR_path', 'var')
            if exist(save_LR_path, 'dir')
                disp(['It will cover ', save_LR_path]);
            else
                mkdir(save_LR_path);
            end
        end

        randn('seed', 0);

        idx = 0;
        filepaths = dir(fullfile(input_path,'*.*'));
        for i = 1 : length(filepaths)
            % randn('seed', 0);
            [paths,imname,ext] = fileparts(filepaths(i).name);
            if isempty(imname)
                disp('Ignore . folder.');
            elseif strcmp(imname, '.')
                disp('Ignore .. folder.');
            else
                idx = idx + 1;
                str_rlt = sprintf('%d\t%s.\n', idx, imname);
                fprintf(str_rlt);

                % read image
                img = im2double(imread(fullfile(input_path, [imname, ext])));

                % add blur
                if kernel_label > 0
                    blurry_img = imfilter(img, double(kernel), 'replicate');
                else
                    blurry_img = img;
                end

                % add noise
                if noise_label > 0
                    noise = noiseSigma/255.*randn(size(blurry_img));
                    im_noise = single(blurry_img + noise);
                    im_noise = im2uint8(im_noise);
                else
                    im_noise = blurry_img;
                end

                if exist('save_LR_path', 'var')
                    imwrite(im_noise, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') file_type]));
                end
            end
        end
    end
end
end
