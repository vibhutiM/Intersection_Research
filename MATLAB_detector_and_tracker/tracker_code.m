%% Motion-Based Multiple Object Tracking (with Data Recording)
%%based on the 'Motion-Based Multiple Object Tracking' tutorial in MATLAB
function multiObjectTracking()
input_name = 'GOPR0118.MP4';
obj = setupSystemObjects(input_name);
tracks = initializeTracks();
old_tracks = initializeTracks();
nextId = 1;
while ~isDone(obj.reader)
    frame = readFrame();
    [centroids, bboxes, mask] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    
    displayTrackingResults();
end
%% Create System Objects
    function obj = setupSystemObjects(input_name)
        obj.reader = vision.VideoFileReader(input_name);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        videoFWriter = vision.VideoFileWriter([input_name, '_tracker_output.avi'],'FrameRate',obj.reader.info.VideoFrameRate);
        % videoFWriter.VideoCompressor='DV Video Encoder';
        obj.videoFWriter=videoFWriter;
        obj.detector = vision.ForegroundDetector('NumGaussians', 25, ...
            'NumTrainingFrames', 600, 'MinimumBackgroundRatio', 0.2);
        
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 30);
    end
%% Initialize Tracks
    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {}, ...
            'old_path', {});
    end
%% Read a Video Frame
    function frame = readFrame()
        frame = obj.reader.step();
    end
%% Detect Objects
    function [centroids, bboxes, mask] = detectObjects(frame)
        
        % Detect foreground.
        mask = obj.detector.step(frame);
        
        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [2,2]));
        mask = imclose(mask, strel('rectangle', [12,12]));
        % mask = imfill(mask, 'holes');
        
        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
    end
%% Predict New Locations of Existing Tracks
    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            
            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end
%% Assign Detections to Tracks
    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end
%% Update Assigned Tracks
    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            correct(tracks(trackIdx).kalmanFilter, centroid);
            tracks(trackIdx).bbox = bbox;
            tracks(trackIdx).old_path = [tracks(trackIdx).old_path;bbox];
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end
%% Update Unassigned Tracks
    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end
%% Delete Lost Tracks
    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        invisibleForTooLong = 1;
        ageThreshold = 8;
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        old_tracks = [old_tracks;tracks(lostInds)];
        tracks = tracks(~lostInds);
    end
%% Create New Tracks
    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        for i = 1:size(centroids, 1)
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
                'old_path', bbox);
            % Add it to the array of tracks.
            tracks = [tracks;newTrack];
            % Increment the next id.
            nextId = nextId + 1;
        end
    end
%% Display Tracking Results
    function displayTrackingResults()
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        x = frame * 0;
        minVisibleCount = 8;
        if ~isempty(tracks)
            reliableTrackInds = [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            if ~isempty(reliableTracks)
                bboxes = cat(1, reliableTracks.bbox);
                %                 for j=1:size(reliableTracks,1)
                %                     reliableTracks(j).old_path = [reliableTracks(j).old_path;bboxes(j,:)];
                %                 end
                ids = int32([reliableTracks(:).id]);
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {''};
                labels = strcat(labels, isPredicted);
                frame = insertObjectAnnotation(frame, 'rectangle',bboxes, labels);
                mask = insertObjectAnnotation(mask, 'rectangle',bboxes, labels);
                x = frame * 0 + 255;
                % x = insertObjectAnnotation(x, 'rectangle',bboxes, labels, 'LineWidth',5);
                bboxes_area = prod(bboxes(:,3:4),2);
                bboxes_cars = bboxes((bboxes_area<=500),:);
                bboxes_pedestrians = bboxes((bboxes_area>500),:);
                x = insertShape(x, 'FilledCircle',[bboxes_pedestrians(:,1)+bboxes_pedestrians(:,3)/2,bboxes_pedestrians(:,2)+bboxes_pedestrians(:,4)/2,(bboxes_pedestrians(:,3)+bboxes_pedestrians(:,4))/4], 'Color', [232,118,46]);
                x = insertShape(x, 'FilledCircle',[bboxes_cars(:,1)+bboxes_cars(:,3)/2,bboxes_cars(:,2)+bboxes_cars(:,4)/2,(bboxes_cars(:,3)+bboxes_cars(:,4))/4], 'Color', [46, 160, 232]);
            end
        end
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
        
        step(obj.videoFWriter, x);
    end
old_tracks = [old_tracks; tracks];
save([input_name '_old_tracks.mat'], 'old_tracks');

load([input_name '_old_tracks.mat']);
A = zeros(1080,1920,3) * 255;
for i=1:length(old_tracks)
    current_track = old_tracks(i).old_path;
    if size(current_track)>1
        sizes = prod(current_track(:,3:4),2);
        mean_size = median(sizes);
        if mean_size<500
            color = 'red';
            color = [232,118,46];
        else
            color = 'blue';
            color = [46, 160, 232];
        end
        for j=1:size(current_track,1)-1
            A = insertShape(A, 'Line', [current_track(j,1)+current_track(j,3)/2, current_track(j,2)+current_track(j,4)/2, current_track(j+1,1)+current_track(j+1,3)/2, current_track(j+1,2)+current_track(j+1,4)/2], 'LineWidth', 9, 'Color', color);
        end
    end
    disp(i/length(old_tracks));
end
imshow(A);
%%
imwrite(A / 255,[input_name '_tracking_image.png']);
end



