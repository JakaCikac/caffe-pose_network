function frcnn_init()
    fprintf('Initializing Faster R-CNN network...\n');

    global frcnn_net_rpn;
    global frcnn_net_det;
    global frcnn_model_info;
    global frcnn_opts;

    addpath('../faster_rcnn/bin');
    addpath('../faster_rcnn/functions/fast_rcnn');
    addpath('../faster_rcnn/functions/nms');
    addpath('../faster_rcnn/functions/rpn');
    addpath('../faster_rcnn/utils');

    frcnn_opts.per_nms_topN         = 6000;
    frcnn_opts.nms_overlap_thres    = 0.7;
    frcnn_opts.after_nms_topN       = 100;
    frcnn_opts.use_gpu              = true;
    frcnn_opts.gpu_id               = 0;
    frcnn_opts.test_scales          = 400;    
    
    model_dir = '../../models/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/';
    %model_dir = '../../models/faster_rcnn_final/faster_rcnn_VOC0712_ZF/';
    
    frcnn_model_info = load(strcat(model_dir, 'model'));
    frcnn_model_info = frcnn_model_info.proposal_detection_model;
	
    frcnn_model_info.conf_proposal.test_scales = frcnn_opts.test_scales;
    frcnn_model_info.conf_detection.test_scales = frcnn_opts.test_scales;
    
	% Region proposal network.
    model_rpn = [model_dir frcnn_model_info.proposal_net];
    model_rpn_def = [model_dir frcnn_model_info.proposal_net_def];
    
    % Detection network.
    model_det = [model_dir frcnn_model_info.detection_net];
    model_det_def = [model_dir frcnn_model_info.detection_net_def];
                            
    if ~exist(model_rpn, 'file') || ~exist(model_rpn_def, 'file') ||...
       ~exist(model_det, 'file') || ~exist(model_det_def, 'file')
        error('Model not found.');
    end
    
    if frcnn_opts.use_gpu,
        caffe.set_mode_gpu();
        caffe.set_device(frcnn_opts.gpu_id);
    else
        caffe.set_mode_cpu();
    end
    
    % Initialize networks.
    frcnn_net_rpn = caffe.Net(model_rpn_def, model_rpn, 'test');
    frcnn_net_det = caffe.Net(model_det_def, model_det, 'test');
end

