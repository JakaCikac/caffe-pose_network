function [im_person, new_labels, crop_bbox] = crop_person(im, labels, bbox, max_jitter_x, max_jitter_y)
% Crop person from the given image and update ground truth accordingly.

    im_size = size(im);
    
    % Expand the person bounding box according to higher dimension.
    if bbox(4) > bbox(3) % If height bigger than width
        person_bbox_size = bbox(4);
        crop_bbox(1) = bbox(1) - (person_bbox_size - bbox(3)) * 0.5;
        crop_bbox(2) = bbox(2);
    else
        person_bbox_size = bbox(3);
        crop_bbox(1) = bbox(1);
        crop_bbox(2) = bbox(2) - (person_bbox_size - bbox(4)) * 0.5;
    end
    crop_bbox(3:4) = person_bbox_size;
    crop_bbox = round(crop_bbox);
    
    % Do jitter.
    if max_jitter_x > 0 || max_jitter_y > 0
       jitter_x = randi(max_jitter_x * 2 + 1) - max_jitter_x - 1;
       jitter_y = randi(max_jitter_y * 2 + 1) - max_jitter_y - 1;
       crop_bbox(1:2) = crop_bbox(1:2) + [jitter_x, jitter_y]; 
    end    
    
    % Define crop area.
    crop_x1 = crop_bbox(1);
    crop_y1 = crop_bbox(2);
    crop_x2 = crop_bbox(1) + crop_bbox(3);
    crop_y2 = crop_bbox(2) + crop_bbox(4);
        
    % Add padding if needed.
    pad_x = max(abs(min(0, crop_x1 - 1)), max(0, crop_x2 - im_size(2)));
    pad_y = max(abs(min(0, crop_y1 - 1)), max(0, crop_y2 - im_size(1)));
    if pad_x > 0 || pad_y > 0
        im_padded = padarray(im, double([pad_y, pad_x]));
    else
        im_padded = im;
    end

    % Cropt the person image.
    offset_x = crop_x1 + pad_x;
    offset_y = crop_y1 + pad_y;
    crop_rect = [offset_x, offset_y, crop_x2 - crop_x1, crop_y2 - crop_y1];
    im_person = imcrop(im_padded, crop_rect);
    
    if ~isempty(labels)
        % Add offset to labels.
        labels_x = labels(1:2:end-1);
        labels_y = labels(2:2:end);
        new_labels_x = round(labels_x + pad_x - offset_x);
        new_labels_y = round(labels_y + pad_y - offset_y);
        new_labels = [new_labels_x', new_labels_y'];
        new_labels = reshape(new_labels.', [1 numel(new_labels)]);
    else
        new_labels = labels;
    end
end

