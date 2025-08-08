load modelparameters.mat

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

Original_image_dir  = "";
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);
niqe = [];
for i = 1:im_num
    IMname = regexp(im_dir(i).name, '\.', 'split');
    IMname = IMname{1};
    im=imread(fullfile(Original_image_dir, im_dir(i).name));
        quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
            mu_prisparam,cov_prisparam);
        niqe = [niqe quality];
end
mNIQE = mean(niqe);
fprintf('The average NIQE = %2.4f. \n', mNIQE);