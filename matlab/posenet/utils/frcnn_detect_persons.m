function [person_boxes, region_proposals] = frcnn_detect_persons(im)
    global frcnn_net_rpn;
    global frcnn_net_det;
    global frcnn_model_info;
    global frcnn_opts;    

    % Test proposal.
    [boxes, scores] = proposal_im_detect(frcnn_model_info.conf_proposal, frcnn_net_rpn, im);
    region_proposals = boxes_filter([boxes, scores], frcnn_opts.per_nms_topN, frcnn_opts.nms_overlap_thres, frcnn_opts.after_nms_topN, frcnn_opts.use_gpu);
        
    % Test detection.
    if frcnn_model_info.is_share_feature == 1,
        [boxes, scores] = fast_rcnn_conv_feat_detect(frcnn_model_info.conf_detection, frcnn_net_det, im, ...
            frcnn_net_rpn.blobs(frcnn_model_info.last_shared_output_blob_name), ...
            region_proposals(:, 1:4), frcnn_opts.after_nms_topN);
    else
        [boxes, scores] = fast_rcnn_im_detect(frcnn_model_info.conf_detection, frcnn_net_det, im, ...
            region_proposals(:, 1:4), frcnn_opts.after_nms_topN);
    end
    
    % Get person boxes only.
    index = 15; % Person index.
    thres = 0.6;
    person_boxes = [boxes(:, (1+(index-1)*4):(index*4)), scores(:, index)];
    person_boxes = person_boxes(nms(person_boxes, 0.3), :);
    I = person_boxes(:, 5) >= thres;
    person_boxes = person_boxes(I, :);
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end