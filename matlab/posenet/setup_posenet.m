function setup_posenet()

    % Build Faster R-CNN.
    
    cur_dir = pwd;
    cd('../faster_rcnn');    
    faster_rcnn_build();
    cd(cur_dir);

    % Download posenet models.
    
    model_dir = '../../models/posenet/';
    
    if ~exist(model_dir, 'dir')
       mkdir(model_dir); 
    end
        
    pretrain_zip_download = 'http://www.ee.oulu.fi/~malinna/files/posenet_pretrain.zip';
    pretrain_zip_file = [model_dir 'posenet_pretrain.zip'];
    
    try
        fprintf('Downloading posenet pretrain model (348 MB). Please wait...\n');
        websave(pretrain_zip_file, pretrain_zip_download);

        fprintf('Unzipping...\n');
        unzip(pretrain_zip_file, model_dir);

        delete(pretrain_zip_file);
        fprintf('Done.\n');
    catch
        fprintf('Error in downloading, please try loading it manually from %s', pretrain_zip_download); 
    end
    
    % Download Faster R-CNN models.
    
    addpath('../faster_rcnn/fetch_data');    
    fetch_faster_rcnn_final_model();
    movefile('../faster_rcnn/output/faster_rcnn_final', '../../models');
    rmdir('../faster_rcnn/output');
end

