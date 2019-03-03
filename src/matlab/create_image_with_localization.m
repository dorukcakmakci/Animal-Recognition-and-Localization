% Creating and opening a .txt file to fetch localization information
%M = csvread('localization_info.txt',0,0,[0,0,99,3]);
localizationFile = fopen('../../data/test/localization_inf.txt', 'r');
line_tmp = fgetl(localizationFile);
M = strsplit(line_tmp, ',');
for i=2:100
    line_tmp = fgetl(localizationFile);
    token_tmp = strsplit(line_tmp, ',');
    M = vertcat(M,token_tmp);
end
fclose(localizationFile);

% Show evaluation results (using pre-defined or interactive boxes)
for imageIndex=0:99
    % Calculating bbxs'
     I = imread(strcat('../../data/test/images/', int2str(imageIndex), '.JPEG'));
    
    f = figure('visible', 'off');
    image_with_text = insertText(I, [1 1], M(imageIndex+1,5), 'FontSize', 18, 'BoxColor', 'green', 'BoxOpacity', 0.4, 'TextColor', 'white');
    imshow(image_with_text);
    hold on;
    rectangle('Position', [str2double(M(imageIndex+1,1)) str2double(M(imageIndex+1,2)) str2double(M(imageIndex+1,3)) str2double(M(imageIndex+1,4))], 'EdgeColor', 'r', 'LineWidth', 3)
    image_name_tmp = strcat('../../img/localization/', int2str(imageIndex), '/localization_plot.png');
    saveas(f, image_name_tmp);
    export_fig(image_name_tmp);
end