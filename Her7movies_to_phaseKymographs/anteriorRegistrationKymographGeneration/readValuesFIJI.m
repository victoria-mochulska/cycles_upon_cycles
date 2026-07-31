
%% reads a text file output by the groovy script
% (used in making kymos from midline)

function values1 = readValuesFIJI(filename)
fileID = fopen(filename,'r');
A = fscanf(fileID,'%s');
A1 = split(A, '"');

nElements = cellfun(@length, A1);
A2 = A1(nElements>0);

%

for ii = 1:numel(A2)
    A3 = A2{ii};
    A4 = split(A3, '[');
    A5 = split(A4{2}, ']');
    A6 = split(A5{1}, ',');
    
    v1 = zeros(1,numel(A6));
    for jj = 1:numel(A6)
        v1(jj) = str2double(A6{jj});
    end
    values1{ii,1} = v1; % distance along midline
end
end
