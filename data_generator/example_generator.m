%% Clean Workspace
clear all; close all; clc;
%% Get Inputs
input_names = dir('*.MP4');
%% Code Beginning:
opticFlow = opticalFlowHS;
%% Transform Training Block (Disabled for Reproducibility)
% frame = step(videoReader);
% imshow(frame);
% [mx,my] = ginputc(4,'ShowPoints',true);
% [sx,sy] = ginputc(4,'ShowPoints',true);
% m = [mx my];
%% Scene Transformation
load transform_m.mat;
sx = [256;256+512;256+512;256];
sy = [256;256;256+512;256+512];
s=[sx sy];
tform = fitgeotrans(m,s,'projective');
%% Problem Setup
bboxes_pedestrians = [];
bboxes_cars = [];
all_blobs = zeros(1,1024,1024,3);
input_frames = zeros(1,1024,1024,3);
%% Have two Blob Analyzers
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 450);
blobAnalysis_2 = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 30, 'MaximumBlobArea', 450);

ik = 1;
%% For Each Video
for video_index=1:length(input_names)
    input_name = input_names(video_index).name;
    ii = 1;
    close all;
	%% Setup Everything We Will Need (Code is Self-Descriptive)
    foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 300);
    videoReader = vision.VideoFileReader(input_name);
    frame = step(videoReader);
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
    %% Begin Processing Frame-by-Frame
    while ~isDone(videoReader)
		%% We want some time difference between examples so that things don't look the same
        if mod(ii,253)~=0
            frame = step(videoReader); 
            frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
            foreground = step(foregroundDetector, frame);
            ii = ii + 1;
        else
			%% Process the first frame
            frame = step(videoReader);
            ii = 1;
            frame = imwarp(frame,tform,'OutputView',imref2d([1024,1024]));
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
            %% Skip 1 second
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
				%% Save everything
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
        end
    end
    all_blobs = all_blobs(2:end,:,:,:);
    input_frames = input_frames(2:end,:,:,:);
    release(videoReader);
end
%% Extra outputs
save(['segmentation_data.mat'],'input_frames','all_blobs','-v7.3');