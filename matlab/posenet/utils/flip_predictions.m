function [new_coords] = flip_predictions(coords, arm_only)
% Horizontally flip the given joint coordinates.

    coords_x = coords(1:2:end-1);
    coords_y = coords(2:2:end);

    coords_x = 1 - coords_x;
    
    new_coords = [coords_x', coords_y'];
    new_coords = reshape(new_coords.', [1 numel(new_coords)]);
    
    % Swap hand and leg coordinates.
    if ~exist('arm_only', 'var')
        indices = [11 12 9 10 7 8 5 6 3 4 1 2 23 24 21 22 19 20 17 18 15 16 13 14 25 26];    
        new_coords = new_coords(indices);        
    end    
end
