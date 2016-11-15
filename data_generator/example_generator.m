
clear all; close all; clc;
%% Options
input_names = dir('*.MP4');
%% Code Beginning:
opticFlow = opticalFlowHS;
% frame = step(videoReader);
% imshow(frame);
% [mx,my] = ginputc(4,'ShowPoints',true);
% [sx,sy] = ginputc(4,'ShowPoints',true);
% m = [mx my];
load transform_m.mat;
sx = [256;256+512;256+512;256];
sy = [256;256;256+512;256+512];
s=[sx sy];
tform = fitgeotrans(m,s,'projective');
% frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
% frame = imtransform(frame,TFORM);
% [y,x] = find(im2bw(Iw,0));
% imshow(frame);
%%
% for i = 1:150
%     frame = step(videoReader); % read the next video frame
%     frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
%     foreground = step(foregroundDetector, frame);
% end

% se = strel('disk', 3);
% mask = imopen(foreground, strel('disk', [2]));
% mask = imclose(mask, strel('disk', [3]));
% filteredForeground = imfill(mask, 'holes');
% % filteredForeground = imopen(foreground, mask);
% figure; imshow(filteredForeground); title('Clean Foreground');
%
% bbox = step(blobAnalysis, filteredForeground);
%
% result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
%
% bbox_2 = step(blobAnalysis_2, filteredForeground);
%
% result = insertShape(result, 'Rectangle', bbox_2, 'Color', 'red');
%
% numCars = size(bbox, 1);
% result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
%     'FontSize', 14);
% figure; imshow(result); title('Detected Objects');
%
%
%
% se = strel('disk', 3); % morphological filter for noise removal
bboxes_pedestrians = [];
bboxes_cars = [];
all_blobs = zeros(1,1024,1024,3);
input_frames = zeros(1,1024,1024,3);
%% Actual Work
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 450);
blobAnalysis_2 = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 30, 'MaximumBlobArea', 450);

ik = 1;
%flows_x = zeros(100,1024,1024);
%flows_y = zeros(100,1024,1024);

for video_index=1:length(input_names)
    input_name = input_names(video_index).name;
    ii = 1;
    close all;
    foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 300);
    videoReader = vision.VideoFileReader(input_name);
    %     videoFWriter = vision.VideoFileWriter([input_name '_detector_output.avi'],'FrameRate',videoReader.info.VideoFrameRate);
    %     videoFWriter.VideoCompressor='DV Video Encoder';
    frame = step(videoReader); % read the next video frame
    frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
    flow = estimateFlow(opticFlow,rgb2gray(frame));
    flows = [];
    flows.Vx = flow.Vx; flows.Vy = flow.Vy;
    flow = flows;
    flow.Vx = double(flow.Vx);
    flow.Vx = flow.Vx - min(flow.Vx(:));
    if max(flow.Vx(:))>0
        flow.Vx = flow.Vx  / max(flow.Vx(:));
    end
    flow.Vy = double(flow.Vy);
    flow.Vy = flow.Vy - min(flow.Vy(:));
    if max(flow.Vy(:))>0
        flow.Vy = flow.Vy  / max(flow.Vy(:));
    end
    foreground = step(foregroundDetector, frame);
    
    %     videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
    %     videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
    while ~isDone(videoReader)
        if mod(ii,253)~=0
            frame = step(videoReader); % read the next video frame
            frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
            foreground = step(foregroundDetector, frame);
            ii = ii + 1;
        else
            frame = step(videoReader); % read the next video frame
            ii = 1;
            frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
%             flow = estimateFlow(opticFlow,rgb2gray(frame));
            foreground = step(foregroundDetector, frame);
            flow = estimateFlow(opticFlow,rgb2gray(frame));
            
            flows = [];
            flows.Vx = flow.Vx; flows.Vy = flow.Vy;
            flow = flows;
            flow.Vx = double(flow.Vx);
            flow.Vx = flow.Vx - min(flow.Vx(:));
            if max(flow.Vx(:))>0
                flow.Vx = flow.Vx  / max(flow.Vx(:));
            end
            flow.Vy = double(flow.Vy);
            flow.Vy = flow.Vy - min(flow.Vy(:));
            if max(flow.Vy(:))>0
                flow.Vy = flow.Vy  / max(flow.Vy(:));
            end
            
            for needless=1:23
                if ~isDone(videoReader)
                    frame2 = step(videoReader); % read the next video frame
                    frame2 = imwarp(frame2,tform,'OutputView',imref2d([1024,1024]));
                    foreground2 = step(foregroundDetector, frame2);
                    flow2 = estimateFlow(opticFlow,rgb2gray(frame2));
                    flows = [];
                    flows.Vx = flow2.Vx; flows.Vy = flow2.Vy;
                    flow2 = flows;
                    flow2.Vx = double(flow2.Vx);
                    flow2.Vx = flow2.Vx - min(flow2.Vx(:));
                    if max(flow2.Vx(:))>0
                        flow2.Vx = flow2.Vx  / max(flow2.Vx(:));
                    end
                    flow2.Vy = double(flow2.Vy);
                    flow2.Vy = flow2.Vy - min(flow2.Vy(:));
                    if max(flow2.Vy(:))>0
                        flow2.Vy = flow2.Vy  / max(flow2.Vy(:));
                    end
                end
                
            end
            if ~isDone(videoReader)
                imwrite(double(flow.Vy) * 1,[num2str(ik) '_train_flow_y.png']);
                imwrite(double(flow2.Vy) * 1,[num2str(ik) '_test_flow_y.png']);
                imwrite(double(flow.Vx) * 1,[num2str(ik) '_train_flow_x.png']);
                imwrite(double(flow2.Vx) * 1,[num2str(ik) '_test_flow_x.png']);
                imwrite(frame,[num2str(ik) '_test_.png']);
                imwrite(double(foreground2),[num2str(ik) '_test_foreground.png']);
                imwrite(frame,[num2str(ik) '_train_.png']);
                imwrite(double(foreground),[num2str(ik) '_train_foreground.png']);
            end
            ik = ik + 1;
            %             step(videoPlayer, blob_labels);  % display the results
        end
    end
    all_blobs = all_blobs(2:end,:,:,:);
    input_frames = input_frames(2:end,:,:,:);
    release(videoReader); % close the video file
end

save(['segmentation_data.mat'],'input_frames','all_blobs','-v7.3');
%%
% save([input_name '_detector_bboxes.mat'],'bboxes_cars','bboxes_pedestrians');
% A = zeros(1080,1920,3);
% for i=1:size(bboxes_pedestrians)
%     current_track = bboxes_pedestrians;
%     color = [232,118,46];
%     for j=1:size(current_track,1)-1
%         A = insertShape(A, 'Line', [current_track(j,1)+current_track(j,3)/2, current_track(j,2)+current_track(j,4)/2, current_track(j+1,1)+current_track(j+1,3)/2, current_track(j+1,2)+current_track(j+1,4)/2], 'LineWidth', 9, 'Color', color);
%     end
% end
%
%
% for i=1:size(bboxes_cars)
%     current_track = bboxes_cars;
%     color = [46, 160, 232];
%     for j=1:size(current_track,1)-1
%         A = insertShape(A, 'Line', [current_track(j,1)+current_track(j,3)/2, current_track(j,2)+current_track(j,4)/2, current_track(j+1,1)+current_track(j+1,3)/2, current_track(j+1,2)+current_track(j+1,4)/2], 'LineWidth', 9, 'Color', color);
%     end
% end
% imwrite(A / 255,[input_image '_detector_output.png']);