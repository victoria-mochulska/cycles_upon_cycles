

%% ------------------------------------------------------------------------------------------------------------------
% Making kymographs
% i) read the files with
% - (a) intensities and coordinates along the midline for the
% her 7 channel.
% - (b) check if the direction of the midline needs to be changed.
% - (c) if it needs to be changed, change it 
% - (d) make both the kymographs and save.

%% -------------------------------------------------------------------------------------------------------------------
%% -------------------------------------------------------------------------------------------------------------------
% specify the folder, sample names, file suffix
meta.masterFolder = '/Volumes/sapna4tb/20200304_medaka/LucData_midlinetest/250914';

midlines_folder = [meta.masterFolder filesep 'midlines/'];
roiValues_folder = [meta.masterFolder filesep 'roiValues/'];% place where distance coordinates and intensities are saved after running groovy script
kymo_folder = [meta.masterFolder filesep 'Kymos/'];

midline_prefix = 'MAX_'; %
midline_suffix = '_midline';

%% -------------------------------------------------------------------------------------------------------------------
%% -------------------------------------------------------------------------------------------------------------------

samplesInfo = dir([midlines_folder midline_prefix '*' midline_suffix '*']);

%%
meta.nSamples = numel(samplesInfo);
kymo_mat_all = cell(1, numel(meta.nSamples));

%
distance_interval = 10; % default. Distance in pixels over which the value is averaged. 
%%
posterior_register = 0;
%% --------------------------------------------------------------------------------------------------------------------
for ii = 1:meta.nSamples
    sampleName_full = samplesInfo(ii).name;
    sampleName_1 = strsplit(sampleName_full, midline_suffix);
    sampleName_1 = sampleName_1{1};
    sampleName_2 = strsplit(sampleName_1, midline_prefix);
    sampleName = sampleName_2{2};
    
    midlineFile = [midlines_folder sampleName_full];
    % get image stats (pixels, timepoints).
    imageInfo = bfopen(midlineFile);
    omeMeta = imageInfo{4};
    pixelsX = double(omeMeta.getPixelsSizeX(0).getValue());
    pixelsY = double(omeMeta.getPixelsSizeY(0).getValue());
    timepoints= size(imageInfo{1},1);
    midline = uint8(zeros(pixelsY, pixelsX, timepoints));

    for jj = 1:timepoints
        midline(:,:,jj) = imread([midlines_folder sampleName_full], jj);
    end

    %%
    distances_midline = readValuesFIJI([roiValues_folder filesep sampleName_1 '_coordinates.csv']);
    intensities_midline = readValuesFIJI([roiValues_folder filesep sampleName_1 '_intensities.csv']);
    %%
    % x and y coordinates of the skeleton.
    coordinates_x = readValuesFIJI([roiValues_folder filesep sampleName_1 '_coordinates_skel_x.csv']);
    coordinates_y = readValuesFIJI([roiValues_folder filesep sampleName_1 '_coordinates_skel_y.csv']);

    %% ----------------------------------------------------------------------------------------------------------------
    %% make intensity kymo
    distance_threshold = ceil(max(distances_midline{end})+20);

    kymo_mat1 = makeKymoWithRoiValues(midline, distances_midline, intensities_midline, ...
        coordinates_x, coordinates_y, distance_interval, distance_threshold, posterior_register);

    kymo_mat_all{ii} = kymo_mat1;
    meta.sampleNames{ii} = sampleName;
end
%% ------------------------------------------------------------------------------------
%% ------------------------------------------------------------------------------------

% save kymographs as .png/.tif files.
%
mkdir(kymo_folder);
%%
for ii = 1:meta.nSamples

    image1 = kymo_mat_all{ii};
    image1 = uint16(image1); 
    image1_smooth = imgaussfilt(image1, 0.5);
    figure; imagesc(image1_smooth); title(strrep(meta.sampleNames{ii}, '_', ' '));
    colorbar; 
    xlabel('timepoints');
    if posterior_register == 1
        ax = gca;
        ax.YTickLabel = flip(ax.YTickLabel);
        suffix = '_posterior';
    else
        suffix = '_anterior';
    end
    
    image1_name = [kymo_folder filesep meta.sampleNames{ii} suffix '_kymo.tif'];
    imwrite(image1_smooth, image1_name);
    
    image1_name = [kymo_folder filesep meta.sampleNames{ii} suffix '_kymo.png'];
    saveas(gcf, image1_name);
end



%% ----------------------------------------------------------------------------------------------------------------
%% ----------------------------------------------------------------------------------------------------------------
%% ----------------------------------------------------------------------------------------------------------------

function kymo_mat = makeKymoWithRoiValues(midline, distances_midline, intensities_midline, ...
    coordinates_x, coordinates_y, distance_interval, distance_threshold, posterior_register)

%% ----------------------------------------------------------------------------------------------
% ---------------input
% 1) midline: 3D array (x,y, t) of binary images with the midline 
% 2) distances_midline: a cell array {t} with distance information along
% the midline for each timepoint (returned by FIJI).
% 3) intensities_midline: a cell array {t} with intensity information along
% the midline for each timepoint (returned by FIJI).
% 4,5) coordinates_x, coordinates_y: a cell array {t} with x and y coordinate information along
% the midline for each timepoint (returned by FIJI).
% 6) distance_interval: 
% 7) distance_threshold: maximum distance (range) over which intensities
% values have to be summed. 
% 8) distance_interval: distance over which intensities values are
% averaged. 
% 9) posterior_register: 0/1 depending on whether you want to register
% posteriorly or not. 
% ---------------output
% 1) kymo_mat1: a 2D array (matrix) with average intensity values along space(r/y) and time(c/x).
%% ----------------------------------------------------------------------------------------------
%%

kymo_mat = zeros(distance_threshold, size(distances_midline,1));

nT = numel(distances_midline);
tic;
for jj = 1:nT
    jj
    i1 = intensities_midline{jj};
    d1 = distances_midline{jj};
    xy = [(coordinates_x{jj})' (coordinates_y{jj})'];

    %% ----------------find out if the intensities values need to be flipped. 
    mask1 = imbinarize(midline(:,:,jj));
    mask1 = imdilate(mask1, strel('disk', 5));
    skel1 = bwskel(mask1);
    skel1 = bwmorph(skel1, 'spur');
    skel1 = bwmorph(skel1, 'shrink');
    endpoints = bwmorph(skel1, 'endpoints');
    
    % get the anterior_posterior ends of the midline
    [endpoints_xy(:,2), endpoints_xy(:,1)] = find(endpoints);

    if (endpoints_xy(1,2)<endpoints_xy(2,2))
        anterior_idx = 1;
    else
        anterior_idx = 2;
    end
    
    posterior_idx = setdiff([1,2], anterior_idx);
    %%
    % determine the orientation of saved intensity profiles (anterior-> posterior or not?)  
    distances_all = pdist2(xy(1,:), endpoints_xy);
    %
    if distances_all(1)<distances_all(2)
        closest_endpoint = 1;
    else
        closest_endpoint = 2;
    end
    
    % 
    if posterior_register == 1
        alignmentPoint = posterior_idx;
    else
        alignmentPoint = anterior_idx;
    end
        

    if closest_endpoint == alignmentPoint
    else
        d1 = max(d1)-d1;
        
    end
    
    %% ---------------------------------------------------------------------

    for kk = 0:distance_threshold-distance_interval
       
        toKeep = [kk, kk+distance_interval];
        idx = d1>toKeep(1) & d1<=toKeep(2);
        d1_new = d1(idx);
        i1_new = i1(idx);

        kymo_mat(kk+1, jj) = mean(nonzeros(i1_new));

    end
end
%
kymo_mat(isnan(kymo_mat))=0;

if posterior_register == 1
    kymo_mat = flipud(kymo_mat);
    d1_new = flipud(d1_new);
end
toc;
end







