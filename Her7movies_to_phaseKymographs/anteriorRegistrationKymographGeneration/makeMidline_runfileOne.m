
%% ------------------------------------------------------------------------------------------
%% NOTE: Before running this code, make sure that
%% tail explants in the images and masks are oriented 
%% such that the anterior (cut side) is placed higher (towards the top) 
%% than the posterior (towards the bottom). 

%% ------------------------------------------------------------------------------------------
%% This file takes in the segmented tail masks and finds the midline for each mask. 
% input: tail masks of one sample  
% output: smooth tail masks, midlines, saved as tiff files
% It currently worksfor one sample at a time. But can easily be scaled up.  


%% ------------------------------------------------------------------------------------
%% ------------------------------------------------------------------------------------
% (1) Specify
% the path to intensity image and mask.
masterFolder = '/Volumes/sapna4tb/20200304_medaka/LucData_midlinetest/250914/';
mkdir([masterFolder '/smoothMasks']);
mkdir([masterFolder '/midlines']);

%%
sampleName = 'MAX_tailex_DE_W0004_P0001';
%%
% input file paths
intensity_image = [masterFolder 'venus_maxprojections/' sampleName '.tif'];
tail_masks = [masterFolder filesep 'segmentation_prediction/' sampleName '_masks.tif'];
%%
% output files
tail_masks_smooth = [masterFolder '/smoothMasks/' sampleName '_smoothMask.tif']; % specify the file name for processed masks
tail_midline = [masterFolder '/midlines/' sampleName '_midline.tif'];
%%

%% ------------------------------------------------------------------------------------
%% ------------------------------------------------------------------------------------
% (2) process tail mask to obtain a smoother mask

% determine the image size
rawImageInfo = bfopen(pathaintensity_image);
omeMeta = rawImageInfo{4};

meta.pixelsX = double(omeMeta.getPixelsSizeX(0).getValue());
meta.pixelsY = double(omeMeta.getPixelsSizeY(0).getValue());

timepoints = 1:size(rawImageInfo{1},1);slack
size(rawImageInfo{1},1)

%%
% initialize a variable to save processed masks
masks_smooth = false(meta.pixelsY, meta.pixelsX, numel(timepoints));

%%
counter = 1;
for ii = timepoints
    mask1 = imread(tail_masks, ii);
    mask1_binary = imbinarize(mask1);
    mask1_clean = imopen(mask1_binary, strel('disk', 5)); % clean
    
    mask1_clean = bwareafilt(mask1_clean,1); % keep only the largest component
    
    window_size = 10; % smooth
    kernel = ones(window_size) / window_size ^ 2;
    blurryImage = conv2(single(mask1_clean), kernel, 'same');
    mask1_smooth = blurryImage > 0.25; % Rethreshold
    
    masks_smooth(:,:,counter) = mask1_smooth;
    counter = counter+1;
end
%%

% % check the smooth masks
figure; hold on;

for ii = timepoints
    imshow(masks_smooth(:,:,ii)); title(num2str(ii));
    pause(0.3);
end
%%

% save the smooth masks
mask1 = uint8(255*masks_smooth(:,:,1));
imwrite(mask1, tail_masks_smooth);

for ii = 2:timepoints(end)
    mask1 = uint8(255*masks_smooth(:,:,ii));
    imwrite(mask1, tail_masks_smooth, 'WriteMode', 'append');
    
end
%% -------------------------------------------------------------------------------
%% -------------------------------------------------------------------------------
% (3) get the midline for the smooth masks.

distance_prune = [30,10]; % [anterior, posterior], trimming the skeleton from the ends before extending it.
% [10,10] is a good starting point. 
%%
midlines = masks_smooth;
%%
% find midline
for ii = timepoints
    mask1 = masks_smooth(:,:,ii);
    midline = getMidline(mask1, distance_prune);
    midline = imdilate(midline, strel('disk', 4)); % if there are weird kinks or spikes, dilating the mask hides them 
    
    midlines(:,:,ii) = midline;
end
%%
% check
figure; hold on;
for ii = timepoints
    mask1 = masks_smooth(:,:,ii);
    midline = midlines(:,:,ii);
    imshowpair(mask1, midline); title(num2str(ii));
    pause(0.4);
end
%%

% if the midline is not correct for some timepoints, re-run the find-
% midline section for those timepoints by modifying the distance_prune
% parameter(increase the values to trim more). 

timepoints_to_correct = [22];
distance_prune = [0,20];

figure; hold on;
for ii = timepoints_to_correct
    mask1 = masks_smooth(:,:,ii);
    midline = getMidline(mask1, distance_prune);
    midline = imdilate(midline, strel('disk', 4));
    
    midlines(:,:,ii) = midline;
    
    imshowpair(mask1, midline); title(num2str(ii));
    pause(0.3);
end

%%
% save midlines.

%%

midline = midlines(:,:,1);
mask1 = uint8(255*midline);
imwrite(mask1, tail_midline);

for ii = 2:timepoints(end)
    midline = midlines(:,:,ii);
    mask1 = uint8(255*midline);
    imwrite(mask1, tail_midline, 'WriteMode', 'append'); 
end

%% -------------------------------------------------------------------------------
%% -------------------------------------------------------------------------------
