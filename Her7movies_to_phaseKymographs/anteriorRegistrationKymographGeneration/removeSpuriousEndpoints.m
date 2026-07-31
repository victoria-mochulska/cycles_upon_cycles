


function [goodEndpoints_image, goodEndpoints_id] = removeSpuriousEndpoints(segment, endpoints, varargin)


%% removeSpuriousEndpoints  Identify the genuine tip endpoints of a
% (possibly branchy) skeleton, discarding spurious endpoints caused by
% short side-spurs near skeleton junctions.
%% -------- previous logic - by me. 
% ---------input---------------- 
% segment: a binary image (skeleton);
% endpoints: a binary image with just the endpoints of the the skeleton

% ---------output---------------- 
% endpoints_new_image: a binary image (skeleton);
% good_endpoints_id: a binary image with just the endpoints of the the skeleton

%%
% endpoints_label = bwlabel(endpoints);
% n_endpoints = numel(unique(endpoints_label)) - 1;
% 
% goodEndpoints_image = endpoints;
% 
% if n_endpoints>2
% 
%     nc = zeros(1, n_endpoints);
%     for ii = 1:n_endpoints
%         e1 = endpoints_label == ii;
%         e1_big = imdilate(e1, strel('disk', 10));
% 
%         segment_new = segment - e1_big;
%         segment_new = imbinarize(segment_new);
%         stats = regionprops(segment_new, 'Area');
%         nc(ii) = numel(stats);
%     end
% 
%     endpoints_to_remove = find(nc > 1);
% 
%     for ii = 1:numel(endpoints_to_remove)
%         endpoints_label(endpoints_label == endpoints_to_remove(ii)) = 0;
%         goodEndpoints_image(endpoints_label == endpoints_to_remove(ii)) = 0;
%     end
% 
%     goodEndpoints_id = find(nc == 1);
% else
% 
%     goodEndpoints_id = 1:2;
% end

%% ------------------------------------------------------------------------------------------
%% ------------------------------------------------------------------------------------------
%% ------------------------------------------------------------------------------------------
%% ------------------------------------------------------------------------------------------


% from Claude: Apparently a better, more fool-proof logic. Haven't tested
% yet. 
%
% ---------- input ----------------
% segment:   binary image (skeleton), logical or 0/1.
% endpoints: binary image with just the candidate endpoints of `segment`
%            (e.g. from bwmorph(segment,'endpoints')).
%
% PV pairs:
% 'nKeep' : number of endpoints to keep. DEFAULT: 2 (matches the
%           assumption made downstream, in pruneSkeleton.m / getMidline.m,
%           that a tail skeleton has exactly two genuine tips).
%
% ---------- output ----------------
% goodEndpoints_image: binary image with exactly nKeep endpoint pixels
%                       retained (the ones judged genuine).
% goodEndpoints_id:    labels (in bwlabel(endpoints) numbering) of the
%                       retained endpoints, sorted ascending.
%
% METHOD
% Each candidate endpoint is scored by its geodesic distance, measured
% along the skeleton, to the nearest branch point (a skeleton pixel with
% 3 or more skeleton-connected neighbors). A spurious endpoint -- the tip
% of a short side branch -- sits close to a branch point. A genuine tip
% of the main path sits far from any branch point (or there may be no
% branch point at all, if the skeleton is a single unbranched curve). The
% nKeep endpoints with the LARGEST such distance are kept.
%
% This replaces an earlier disk-dilation + connectivity-after-removal
% heuristic. That version had two problems this version fixes:
%   (1) no guarantee that exactly nKeep endpoints would survive -- it was
%       possible to end up with 0, 1, 3+ "good" endpoints, which would
%       then crash the caller (pruneSkeleton.m indexes row 2 of the
%       result unconditionally).
%   (2) it relied on `imbinarize` to clean up a signed image produced by
%       subtracting a dilated mask from a binary skeleton (segment -
%       e1_big can be -1), which works by luck rather than by being the
%       intended operation. Logical masking (& ~mask) is used here
%       instead, wherever an equivalent step is needed.
%
% See also: pruneSkeleton, longestConstrainedPath, bwdistgeodesic

%% ---- parse inputs ----
p = inputParser;
p.addParameter('nKeep', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.parse(varargin{:});
nKeep = p.Results.nKeep;

segment   = logical(segment);
endpoints = logical(endpoints) & segment;  % guard against mismatched inputs

endpoints_label = bwlabel(endpoints);
n_endpoints = max(endpoints_label(:));

if n_endpoints == 0
    error('removeSpuriousEndpoints:noEndpoints', ...
        'No endpoints found in the input skeleton; cannot proceed.');
end

if n_endpoints <= nKeep
    % Nothing to prune -- already at or below the target count.
    % (Matches the original function's early return for n_endpoints<=2,
    % generalized to nKeep.)
    goodEndpoints_image = endpoints;
    goodEndpoints_id = 1:n_endpoints;
    return
end

%% ---- find branch points: skeleton pixels with >=3 skeleton-neighbors ----
neighborCount = conv2(double(segment), ones(3), 'same') - double(segment);
branchPoints = segment & (neighborCount >= 3);

if ~any(branchPoints(:))
    % More than nKeep endpoints, yet no single pixel has 3+ neighbors.
    % This happens if `segment` is actually several disconnected simple
    % curves rather than one branching tree (each fragment contributing
    % its own 2 endpoints, with no true junction anywhere). There's no
    % branch-point geometry to score against, so genuine-vs-spurious
    % can't be determined this way. Fall back to keeping the first
    % nKeep (by label order) and warn loudly, rather than erroring or
    % guessing silently.
    warning('removeSpuriousEndpoints:noJunctions', ...
        ['Found %d endpoints but no skeleton junction pixels to ', ...
         'distinguish genuine tips from spurs (likely disconnected ', ...
         'skeleton fragments). Keeping the first %d endpoints by label.'], ...
        n_endpoints, nKeep);
    keepLabels = 1:nKeep;
else
    %% ---- geodesic distance from every skeleton pixel to nearest branch point ----
    distToBranch = bwdistgeodesic(segment, branchPoints, 'quasi-euclidean');

    %% ---- score each endpoint by its distance to the nearest branch point ----
    score = nan(1, n_endpoints);
    for ii = 1:n_endpoints
        idx = find(endpoints_label == ii);
        d = distToBranch(idx);
        d = d(isfinite(d));
        if isempty(d)
            % Endpoint pixel is geodesically disconnected from every
            % branch point (e.g. it sits on its own disconnected
            % fragment). Treat it as maximally genuine rather than
            % discarding it by accident.
            score(ii) = Inf;
        else
            score(ii) = d(1);
        end
    end

    %% ---- keep the nKeep endpoints farthest from any junction ----
    [~, sortIdx] = sort(score, 'descend');
    keepLabels = sortIdx(1:nKeep);
end

goodEndpoints_image = ismember(endpoints_label, keepLabels) & endpoints;
goodEndpoints_id = sort(keepLabels);

end






















