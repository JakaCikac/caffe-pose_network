function [predictions, person_boxes, net_time] = posenet_predict(im, person_centered)

global posenet_net;

if ~exist('person_centered', 'var')
    person_centered = false;
end

posenet_frame_size = [224 224];
num_predictions = 26;

im_size = size(im);

if ~person_centered
    % Get person bounding boxes.
    tic
    bboxes = frcnn_detect_persons(im);
    frcnn_time = toc;
    
    num_detected_persons = size(bboxes, 1);
else
    num_detected_persons = 1;
    frcnn_time = 0;
end

person_boxes = zeros(num_detected_persons, 5);
predictions = zeros(num_detected_persons, num_predictions);

posenet_time = 0;

for i = 1:num_detected_persons
    if ~person_centered
        % Crop the person image.
        person_box = bboxes(i,:);
        person_box(3) = min(person_box(3), im_size(2)) - person_box(1); % Width
        person_box(4) = min(person_box(4), im_size(1)) - person_box(2); % Height

        [im_person, ~, crop_bbox] = crop_person(im, [], person_box(1:4), 0, 0);
        person_boxes(i,:) = person_box;
    else
        im_person = im; 
        crop_bbox = [1 1 posenet_frame_size];
        person_boxes(i,:) = zeros(1,5);
    end
    
    im_person_size = size(im_person);
    
    % Convert an image returned by Matlab's imread to im_data in caffe's data
    % format: W x H x C with BGR channels
    im_data = im_person(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, posenet_frame_size);  % resize im_data
    im_data = im_data - 127; % Reduce pixel mean value.
    im_data = im_data * 0.0078125; % Normalize pixel values to range [-1,1].
    
    % Predict joint positions.
    net_data = zeros([posenet_frame_size 3 2], 'single');
    net_data(:,:,:,1) = im_data;   
    net_data(:,:,:,2) = flipud(im_data);   
    tic
    net_result = posenet_net.forward({net_data});   
    posenet_time = posenet_time + toc;
    net_result = net_result{1}';
    predictions_1 = net_result(1,:);
    predictions_2 = flip_predictions(net_result(2,:));
    
    % Final predictions is average of the original and flipped image 
    % predictions.
    predictions(i,:) = (predictions_1 + predictions_2) * 0.5;
    predictions(i,1:2:end) = predictions(i,1:2:end) * im_person_size(2) + crop_bbox(1);
    predictions(i,2:2:end) = predictions(i,2:2:end) * im_person_size(1) + crop_bbox(2);
end

net_time = frcnn_time + posenet_time;