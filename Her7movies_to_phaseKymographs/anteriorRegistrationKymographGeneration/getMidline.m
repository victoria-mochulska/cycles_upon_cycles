function midline = getMidline(mask, distance_prune)
%% ------ extrapolate the skeleton on either side to obtain the midline of
% the tail---------------------------

%% input: binary mask of the tail
%% output: binary mask of the midline of the tail
%% ---------------------------------------------------------------------------
%% ---------------------------------------------------------------------------
% 1) get the two endpoints of the skeleton (cartesian coordiantes)
% 2) for each endpoint, extract the region on the skelton nearby
% (distance < distance_max; subskeleton)
% 3) find the direction in which to extrapolate the sub-skeleton
% 4) extrapolate (+idx_extn pixels)
% 5) Convert cartesian coordinates into image coordinates. 
%% ---------------------------------------------------------------------------
%% ---------------------------------------------------------------------------
if ~exist('distance_prune', 'var')
    distance_prune = [5 5];
end

distance_max = 20; %[10 works well for 256*256 image].
idx_extn = 200; % [50 works well for 256*256 image].
n_extn = 2*idx_extn; 
%% ---------------------------------------------------------------------------
n_pixels_x = size(mask,2); 
n_pixels_y = size(mask,1);

r_fit_all = zeros(2, n_extn);
c_fit_all = r_fit_all;

%% ---------------------------------------------------------------------------
%(1)
% (i)find the skeleton
skel1 = bwskel(mask);

%(ii)process the skeleton
% (a) keep the longest branch, if there are is more than one branch in the
% skeleton.
% (b) trim it. 

skel1 = longestConstrainedPath(skel1);
skel1_prune = pruneSkeleton(skel1, distance_prune);
%
%(iii)get endpoints
endpoints = bwmorph(skel1_prune, 'endpoints');

endpoints_label = bwlabel(endpoints);
stats = regionprops(endpoints_label, 'Centroid');
endpoints_rc = fliplr(cat(1, stats.Centroid)); % corrdinates of the two endpoints


%% ---------------------------------------------------------------------------
%% ---------------------------------------------------------------------------
% 2)
skel2 = skel1_prune;

for kk = [1,2]
    endpoints_skel1_image = endpoints_label == kk;
    endpoints_skel1 = endpoints_rc(kk,:);
    
    distance_1 = double(bwdist(endpoints_skel1_image)).*double(skel1_prune);
    segment = (distance_1<distance_max).*skel1_prune; % to extend

    %%
    endpoints_segment_image = bwmorph(segment, 'endpoints');
    [endpoints_segment(:,1), endpoints_segment(:,2)] = find(endpoints_segment_image);
    %figure; imshowpair(skel1, distance_1_new);
    
    skel2 = skel2&~(segment);
    %%
    %(3)
    % index of the closest endpoint
    distance_from_old_endpoint = pdist2(endpoints_segment, endpoints_skel1);
    if abs(distance_from_old_endpoint(1)) < abs(distance_from_old_endpoint(2))
        segment_idx_close = 1;
        segment_idx_far = 2;
    else
        segment_idx_close = 2;
        segment_idx_far = 1;
    end
    %%
    
    % equation of the line passing through the segment
    [r, c] = find(segment);
    coeff = polyfit(c, r, 1);
    slope = coeff(1);
    
    %%
    %(3,4)
    if abs(slope) > 100000 || numel(unique(c)) < 3 % when line is x = constant
        c_fit = repmat(mode(c), 1, n_extn);
        
        if endpoints_segment(segment_idx_close,1) - endpoints_segment(segment_idx_far,1) < 0 % find direction of extrapolation
            r_fit = linspace(endpoints_segment(segment_idx_far,1)-idx_extn, endpoints_segment(segment_idx_far,1), n_extn);
        else
            r_fit = linspace(endpoints_segment(segment_idx_far,1), endpoints_segment(segment_idx_far,1)+idx_extn, n_extn);
        end
        
    else
        
        if endpoints_segment(segment_idx_close, 2) - endpoints_segment(segment_idx_far, 2) < 0
            c_fit = linspace(endpoints_segment(segment_idx_far,2)-idx_extn, endpoints_segment(segment_idx_far,2), n_extn);
        else
            c_fit = linspace(endpoints_segment(segment_idx_far,2), endpoints_segment(segment_idx_far,2)+idx_extn, n_extn);
        end
        r_fit = polyval(coeff, c_fit,1);
    end
    
    r_fit_all(kk,:) = r_fit;
    c_fit_all(kk,:) = c_fit;
    
    %%
       %figure; plot(c, -r, 'b-'); hold on;
        %  plot(c_fit, -r_fit, 'r-');
    %     % %    
end

%% ----------------------------------------------------------------
%% ----------------------------------------------------------------

% (5)
c_fit = round(c_fit_all);
r_fit = round(r_fit_all);

idx = c_fit>0 & c_fit<=n_pixels_x & r_fit>0 & r_fit<=n_pixels_y;

rFit_new = r_fit(idx);
cFit_new = c_fit(idx);

%%
skel3 = zeros(size(skel1_prune));
for kk = 1:numel(rFit_new)
    skel3(rFit_new(kk), cFit_new(kk)) = 1;
end

skel3 = logical(skel3)|skel2;
skel4 = skel3.*mask;
%%

if numel(unique(bwlabel(skel4))) > 2
    skel4  = bwmorph(skel4, 'bridge');
end
midline = skel1_prune|skel4;

%% ----------------------------------------------------------------------------------------------------------
end






