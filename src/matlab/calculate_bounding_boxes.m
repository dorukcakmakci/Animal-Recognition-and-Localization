% Load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

% Set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .80;     % step size of sliding window search
opts.beta  = .50;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 50;  % max number of boxes to detect

% Creating and opening a .txt file to save localization information
localizationFile = fopen('../../data/test/bbx_information.txt', 'wt');

% Show evaluation results (using pre-defined or interactive boxes)
for imageIndex=0:99
    % Calculating bbxs'
    I = imread(strcat('../../data/test/images/', int2str(imageIndex), '.JPEG'));
    bbs=edgeBoxes(I,model,opts);
    % Creating a directory for the corresponding image
    mkdir(strcat('../../data/test/windows/', int2str(imageIndex)));
    % Writing the filename to bbx_information.txt
    fprintf(localizationFile, '%s:', int2str(imageIndex));
    for i=1:50
        croppedImage = imcrop(I, [bbs(i,1) bbs(i,2) bbs(i,3) bbs(i,4)]);
        imwrite(croppedImage, strcat('../../data/test/windows/', int2str(imageIndex), '/', int2str(i), '.JPEG'));
        fprintf(localizationFile, '%f,%f,%f,%f;', bbs(i,1), bbs(i,2), bbs(i,3), bbs(i,4));
    end
    % Writing localization information
    fprintf(localizationFile, '\n');
    
    f = figure('visible', 'off');
    imshow(I);
    hold on;
    for i=1:50
        rectangle('Position', [bbs(i,1) bbs(i,2) bbs(i,3) bbs(i,4)], 'EdgeColor', rand(1,3), 'LineWidth', 3)
    end
    % Creating a directory for the corresponding image
    mkdir(strcat('../../img/localization/', int2str(imageIndex)));
    image_name_tmp = strcat('../../img/localization/', int2str(imageIndex), '/bbx_plot.png');
    saveas(f, image_name_tmp);
    export_fig(image_name_tmp);
end
fclose(localizationFile);