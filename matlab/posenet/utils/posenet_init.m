function posenet_init(use_gpu, net_file, prototxt_file, predict_person_centered_images)
    
fprintf('Initializing Posenet network...\n');

model_dir = '../../models/posenet/';

gpu_id = 0;

if ~exist('use_gpu', 'var')
    use_gpu = true;
end

if ~exist('net_file', 'var')
    net_file = 'posenet_pretrain.caffemodel';
end

if ~exist('prototxt_file', 'var')
    model_file = 'posenet_pretrain.prototxt';
else
    model_file = prototxt_file;
end

if ~exist('predict_person_centered_images', 'var')
    predict_person_centered_images = false;
end

% Set caffe mode.
if use_gpu,
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

% Initialize the network.
net_model = [model_dir model_file];
net_weights = [model_dir net_file];

phase = 'test'; % run with phase test (so that dropout isn't applied)

if ~exist(net_weights, 'file')
	error('Model not found.');
end

global posenet_net;

posenet_net = caffe.Net(net_model, net_weights, phase);

% Initialize person detector.
if ~predict_person_centered_images
    frcnn_init();
end

