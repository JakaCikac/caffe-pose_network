function demo_pretrain()

    image_path = '../../data/posenet/demo/';

    use_gpu = true;    
    
    configurations{1}.net = 'posenet_pretrain.caffemodel';
    configurations{1}.model = 'posenet_pretrain.prototxt';

    c = configurations{1};

    rng('shuffle');
    
    % Add caffe/matlab to you Matlab search PATH to use matcaffe
    if exist('../+caffe', 'dir')
      addpath('..');
    else
      error('Please run this script from matlab/posenet');
    end

    addpath(strcat(pwd, '/utils'));

    images = dir([image_path '*.jpg']);
    num_images = length(images);

    % Initialize posenet.
    posenet_init(use_gpu, c.net, c.model, false);

    fprintf('Found %d images.\n', num_images);
    
    for i = 1:num_images
        image_file = [image_path images(i).name];
        
        fprintf('Testing image %d: %s\n', i, image_file);
        
        im = imread(image_file);

        [predictions, person_boxes, net_time] = posenet_predict(im, false);

        num_detected_persons = size(predictions, 1);
        num_predictions = size(predictions, 2);           
        weights = ones(1, num_predictions);
        
        imshow(im);

        title_str = sprintf('%s, detected persons: %d, net time: %.0fms',...
            images(i).name, num_detected_persons, net_time*1000);
        title(title_str);
        axis off
        hold on

        for j = 1:num_detected_persons
            bbox = person_boxes(j,1:4);
            rectangle('Position', bbox, 'EdgeColor', [0 1 0], 'LineWidth', 4);
            
            text(double(bbox(1)), double(bbox(2)), ...
                sprintf('%.2f', double(person_boxes(j,5))), ...
                'FontSize', 14, 'Color', [1 1 1], ...
                'VerticalAlignment', 'bottom', 'BackgroundColor', 'blue');            
            
            draw_skeleton(predictions(j,:), weights, [0 1 0]);
        end        
        
        hold off
        
        try
            k = waitforbuttonpress;
            if k == 1,
                key = get(gcf, 'CurrentKey');
                if strcmp(key, 'escape')
                    break
                end
            end
        catch
            break
        end        
    end
    
    % Close the figure.
    close();
    
    % Clean caffe.
    caffe.reset_all();            
end