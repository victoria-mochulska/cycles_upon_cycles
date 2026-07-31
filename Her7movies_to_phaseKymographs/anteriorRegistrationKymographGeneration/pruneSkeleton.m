
function skeleton_pruned = pruneSkeleton(skeleton, distance_prune)
%% returns a skeleton pruned (cut) by a distance (distance_prune) on both ends.
%% prevents extending the skeleton in erroneous directions.


skel1 = skeleton;
skel2 = skel1;

endpoints = bwmorph(skel1, 'endpoints');
[goodEndpoints_image, ~] = removeSpuriousEndpoints(skel1, endpoints);

%% find the anterior and posterior endpoints (they have different pruning)
skeleton_endpoints_label = bwlabel(goodEndpoints_image);
[skeleton_endpoints_rc(:,1), skeleton_endpoints_rc(:,2)] = find(goodEndpoints_image);
if skeleton_endpoints_rc(1,1) <  skeleton_endpoints_rc(2,1)
    
    skeleton_endpoints_anterior = skeleton_endpoints_label(skeleton_endpoints_rc(1,1),skeleton_endpoints_rc(1,2));
else
    skeleton_endpoints_anterior = skeleton_endpoints_label(skeleton_endpoints_rc(2,1),skeleton_endpoints_rc(2,2));
end
skeleton_endpoints_posterior = setxor([1,2],  skeleton_endpoints_anterior);

%%
counter = 1;
for kk = [skeleton_endpoints_anterior, skeleton_endpoints_posterior]
    endpoints_skel1_image = skeleton_endpoints_label == kk;
    distance_1 = double(bwdist(endpoints_skel1_image)).*double(skel1);
    segment = (distance_1<distance_prune(counter)).*skel1; % to extend
    counter = counter+1;
    %%
    skel2 = skel2&~segment;
end

%%
skeleton_pruned = skel2;
end