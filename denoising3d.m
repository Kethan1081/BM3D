sigma=0.1225; window_size= 8; search_width= 19; 
l2= 0; selection_number = 8; l3= 2.7;

org_img = (imread('lena.jpg'));
org_img = org_img(500:900,470:770, :);
org_img = imresize(org_img,[256-search_width*2,256-search_width*2]);
imwrite(org_img,'results/input_image.jpg');

for channel = 1:3
    img = padarray(org_img(:,:,channel),[search_width search_width], ...
        'symmetric','both');
    noisy_image = imnoise(img,'gaussian', 0, sigma*sigma);
    noisy_img(:,:,channel) = noisy_image;
    basic_result(:,:,channel) = block_matching(noisy_image, sigma, ...
        window_size, search_width, l2, l3, selection_number);
    basic_padded = padarray(basic_result(:,:,channel), ...
        [search_width search_width],'symmetric','both');
    final_result(:,:,channel) = wiener_filter(noisy_image,basic_padded, ...
        sigma, window_size, search_width, l2, selection_number);
end
noisy_img = noisy_img(search_width+1:end-search_width, ...
    search_width+1:end-search_width,:);
imwrite(noisy_img,'results/noisy_image.jpg');
imwrite(uint8(basic_result),'results/res_phase1.jpg');
imwrite(uint8(final_result),'results/res_phase2.jpg');
img = org_img;
mypsnr(1) = psnr(uint8(double(noisy_img)*mean(img(:)) ... 
    /mean(noisy_img(:))),img);
mypsnr(2) = psnr(uint8(final_result*mean(img(:)) ...
    /mean(final_result(:))),(img));
disp(mypsnr);

function basic_result = block_matching(noisy_image, sigma, ws, sw, l2, l3, sn)
    image_size = size(noisy_image);
    numerator = zeros(size(noisy_image));
    denomerator = zeros(size(noisy_image));
    bpr = (sw*2+1) - ws + 1; % number of blocks per row/col
    center_block = bpr^2 / 2 + bpr/2 + 1;
    for i = sw+1:image_size(1)-sw
        for j = sw+1:image_size(2)-sw
            window = noisy_image(i-sw:i+sw , j-sw:j+sw);
            blocks = double(im2col(window, [ws ws], 'sliding'));
            dist = zeros(size(blocks,2),1);
            for k = 1:size(blocks,2)
                tmp = wthresh(blocks(:,center_block),'h',sigma*l2)- ...
                    wthresh(blocks(:,k),'h',sigma*l2);
                tmp = reshape(tmp, [ws ws]);
                tmp = norm(tmp,2)^2;
                dist(k) = tmp/(ws^2);
            end
            [~, inds] = sort(dist);
            inds = inds(1:sn);
            blocks = blocks(:, inds);
            p = zeros([ws ws sn]);
            for k = 1:sn
                p(:,:,k) = reshape(blocks(:,k), [ws ws]);
            end
            p = trans3d(p);
            p = wthresh(p,'h',sigma*l3);
            wp = 1/sum(p(:)>0);
            p = trans3d(p,'inverse');
            for k = 1:sn
                x = max(1,i-sw) + floor((center_block-1)/bpr);
                y = max(1,j-sw) + (mod(center_block-1,bpr));
                numerator(x:x+ws-1 , y:y+ws-1) = ...
                    numerator(x:x+ws-1 , y:y+ws-1) + (wp * p(:,:,k));
                denomerator(x:x+ws-1 , y:y+ws-1) = ...
                    denomerator(x:x+ws-1 , y:y+ws-1) + wp;
            end
        end
    end 
    basic_result = numerator./denomerator;
    basic_result = basic_result(sw+1:end-sw,sw+1:end-sw);
end


function result = wiener_filter(noisy_image, basic_res, sigma, ws,sw, l2, sn)
    image_size = size(noisy_image);
    numerator = zeros(size(noisy_image));
    denomerator = zeros(size(noisy_image));
    bpr = (sw*2+1) - ws + 1; % number of blocks per row/col
    center_block = bpr^2 / 2 + bpr/2 + 1;
    for i = sw+1:image_size(1)-sw
        for j = sw+1:image_size(2)-sw
            window  = noisy_image (i-sw:i+sw, j-sw:j+sw);
            window2 = basic_res(i-sw:i+sw, j-sw:j+sw);
            blocks  = double(im2col(window, [ws ws], 'sliding'));
            blocks2 = double(im2col(window2, [ws ws], 'sliding'));
            dist = zeros(size(blocks,2),1);
            for k = 1:size(blocks,2)
                tmp = wthresh(blocks2(:,center_block),'h',sigma*l2)- ...
                    wthresh(blocks2(:,k),'h',sigma*l2);
                tmp = reshape(tmp, [ws ws]);
                tmp = norm(tmp,2)^2;
                dist(k) = tmp/(ws^2);
            end
            [~, I] = sort(dist);
            I = I(1:sn);
            blocks = blocks(:, I);
            blocks2 = blocks2(:, I);
            p = zeros([ws ws sn]);
            basic_p = zeros([ws ws sn]);
            for k = 1:sn
                p(:,:,k) = reshape(blocks(:,k), [ws ws]);
                basic_p(:,:,k) = reshape(blocks2(:,k), [ws ws]);
            end
            basic_p = trans3d(basic_p);
            wp = zeros(sn,1);
            for k = 1:sn
                tmp = basic_p(:,:,k);
                tmp = norm(tmp,1).^2;
                wp(k) = tmp/(tmp+(sigma^2));
            end
            p = trans3d(p);
            for k = 1:sn
                p(:,:,k) = p(:,:,k)*wp(k);
            end
            p = trans3d(p,'inverse');
            wp = 1/sum(wp(:).^2);
            for k = 1:sn
                q = p(:,:,k);
                x = max(1,i-sw) + floor((center_block-1)/bpr);
                y = max(1,j-sw) + (mod(center_block-1,bpr));
                numerator(x:x+ws-1 , y:y+ws-1) = ...
                    numerator(x:x+ws-1 , y:y+ws-1) + (wp * q);
                denomerator(x:x+ws-1 , y:y+ws-1) = ...
                    wp + denomerator(x:x+ws-1 , y:y+ws-1);
            end
        end
    end 
    result = numerator./denomerator;
    result = result(sw+1:end-sw,sw+1:end-sw);
end

function res = trans3d(data, mode)
if(nargin > 2 && mode == 'inverse')
    res = idct(data,'dim',3);
    for i = 1:size(data,3)
        res(:,:,i)=idct2(res(:,:,i));
    end
else
    res=zeros(size(data));
    for i=1:size(data,2)
        res(:,:,i)=dct2(data(:,:,i));
    end
    dct(res,'dim',3);
end
end