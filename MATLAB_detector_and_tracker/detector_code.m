%% Motion-Based Multiple Object Tracking (with Data Recording)
%%based on the 'Detecting Cars Using Gaussian Mixture Models' tutorial in MATLAB
clear all; close all; clc;
%% Options
input_name = 'GOPR0118.MP4';
%% Code Beginning:
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 300);
videoReader = vision.VideoFileReader(input_name);
videoWriter = VideoWriter([input_name '_detector_output.avi'],'MPEG-4');
open(videoWriter);
for i = 1:1
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end

se = strel('disk', 3);
filteredForeground = imopen(foreground, se);
figure; imshow(filteredForeground); title('Clean Foreground');

blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 150);
bbox = step(blobAnalysis, filteredForeground);

result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

blobAnalysis_2 = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 30, 'MaximumBlobArea', 90);
bbox_2 = step(blobAnalysis_2, filteredForeground);

result = insertShape(result, 'Rectangle', bbox_2, 'Color', 'red');

numCars = size(bbox, 1);
result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
    'FontSize', 14);
figure; imshow(result); title('Detected Objects');


videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
se = strel('square', 3); % morphological filter for noise removal
%% Actual Work
bboxes_pedestrians = [];
bboxes_cars = [];
while ~isDone(videoReader)
    
    frame = step(videoReader); % read the next video frame
    
    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);
    
    % Use morphological opening to remove noise in the foreground
    filteredForeground = imopen(foreground, se);
    
    bbox = step(blobAnalysis, filteredForeground);
    bboxes_cars = [bboxes_cars; bbox];
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
    bbox = step(blobAnalysis_2, filteredForeground);
    bboxes_pedestrians = [bboxes_pedestrians; bbox];
    result = insertShape(result, 'Rectangle', bbox, 'Color', 'red');
    % Display the number of cars found in the video frame
    numCars = size(bbox, 1);
    result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);
    
    step(videoPlayer, result);  % display the results
    writeVideo(videoWriter, result);
end

release(videoReader); % close the video file
close(videoWriter);
%%
save([input_name '_detector_bboxes.mat'],'bboxes_cars','bboxes_pedestrians');
disp('Beginning to write the image...');
A = zeros(1080,1920,3);
% for i=1:size(bboxes_pedestrians,1)
current_track = bboxes_pedestrians;
color = [232,118,46];
%     for j=1:size(current_track,1)-1
%         A = insertShape(A, 'Line', [current_track(j,1)+current_track(j,3)/2, current_track(j,2)+current_track(j,4)/2, current_track(j+1,1)+current_track(j+1,3)/2, current_track(j+1,2)+current_track(j+1,4)/2], 'LineWidth', 9, 'Color', color);
%     end
%A = insertShape(A, 'Line', [current_track(j,1)+current_track(j,3)/2, current_track(j,2)+current_track(j,4)/2, current_track(j+1,1)+current_track(j+1,3)/2, current_track(j+1,2)+current_track(j+1,4)/2], 'LineWidth', 9, 'Color', color);
A = insertShape(A, 'FilledCircle',[bboxes_pedestrians(:,1)+bboxes_pedestrians(:,3)/2,bboxes_pedestrians(:,2)+bboxes_pedestrians(:,4)/2,(bboxes_pedestrians(:,3)+bboxes_pedestrians(:,4))/25], 'Color', [232,118,46]);
disp(i/size(bboxes_pedestrians,1)/2);
% end


% for i=1:size(bboxes_cars,1)
current_track = bboxes_cars;
color = [46, 160, 232];
%     for j=1:size(current_track,1)-1
%         A = insertShape(A, 'Line', [current_track(j,1)+current_track(j,3)/2, current_track(j,2)+current_track(j,4)/2, current_track(j+1,1)+current_track(j+1,3)/2, current_track(j+1,2)+current_track(j+1,4)/2], 'LineWidth', 9, 'Color', color);
%     end
A = insertShape(A, 'FilledCircle',[bboxes_cars(:,1)+bboxes_cars(:,3)/2,bboxes_cars(:,2)+bboxes_cars(:,4)/2,(bboxes_cars(:,3)+bboxes_cars(:,4))/25], 'Color', [46, 160, 232]);
disp(i/size(bboxes_cars,1)/2+0.5);
% end
imshow(A/255);
imwrite(A / 255,[input_name '_detector_output.png']);