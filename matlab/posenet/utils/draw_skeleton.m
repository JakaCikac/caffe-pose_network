function draw_skeleton(coords, weights, color)
    
    num_points = length(coords) / 2;  
    i_offset = 0;
    if num_points == 7 % Upper body
        coords = [zeros(1, 12) coords];
        weights = [zeros(1, 12) weights];
        i_offset = 6;
    elseif num_points == 4 % Arm + head
        coords = [zeros(1, 18) coords];
        weights = [zeros(1, 18) weights];
        i_offset = 9;
    elseif num_points == 3 % Arm
        coords = [zeros(1, 18) coords 0 0];
        weights = [zeros(1, 18) weights 0 0];
        i_offset = 9;
    end

    hold on
    
    % Legs
    draw_connected_line(coords(1:12), weights(1:12), color * 0.8);
    
    % Arms
    draw_connected_line(coords(13:24), weights(13:24), color * 0.8);
    
    % Torso
    if all(weights(17:20))
        shoulder_center = (coords(17:18) + coords(19:20)) * 0.5;        
        draw_connected_line([shoulder_center coords(25:26)], [1 1 weights(25:26)], color * 0.8);
        
        if all(weights(5:8))
            hip_center = (coords(5:6) + coords(7:8)) * 0.5;
            draw_connected_line([shoulder_center hip_center], [1 1 1 1], color * 0.8);
        end        
    end
        
    % Draw ground truth points.
    num_points = length(coords) / 2; 
    for i = 1:num_points
        j = (i-1)*2+1;        
        if sum(weights(j:j+1)) == 2
            x = coords(j);
            y = coords(j+1);
            plot(x, y, ...
                'Marker', 'o', 'MarkerFaceColor', color, ...
                'MarkerSize', 7, 'Color', color);

            text(double(x)+1, double(y)+1, sprintf('%d', i-i_offset), 'FontSize', 14, 'Color', color, 'VerticalAlignment', 'top');
        end
    end
    
    hold off
end

function draw_connected_line(coords, weights, color)
    num_lines = length(coords)/2 - 1;
    for j = 1:num_lines
        i = j*2-1;
        if sum(weights(i:i+3)) == 4
            px = coords([i, i+2]);
            py = coords([i+1, i+3]);
            
            line(px, py, 'LineWidth', 2, 'Color', color);
        end
    end
end
