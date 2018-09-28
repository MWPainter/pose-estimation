import sys
import os
import cv2
from concurrent.futures import ThreadPoolExecutor


# Tell people how to use this
if len(sys.argv) != 4:
    print("Usage: 'python extract_video_frames_script <human_36m_videos_base_dir> <output_dir> <num_workers>'.")
    quit()


# Unpack args
base_dir = sys.argv[1]
output_dir = os.path.join(sys.argv[2], "VideoFrames")
num_workers = int(sys.argv[3])


# Make our output directory
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


# Function to take a video filen, and then output all of the frames to the output directory
def process_video(subject, action, camera, video_filename, video_output_dir):
    # Print progress
    print("Starting unpacking subject '{s}', action '{a}', camera '{c}':".format(s=subject, a=action, c=camera))

    # Open the video, and iterate through the frames
    vid = cv2.VideoCapture(video_filename)
    frame = 0
    success, img = vid.read()
    while success:
        # Make our output filename <output_dir>/subject.action.camera.frame.jpg
        out_filename = os.path.join(video_output_dir, "{f}.jpg".format(f=frame))
        cv2.imwrite(out_filename, img)

        # Get the next frame
        frame += 1
        success, img = vid.read()

        # Progress
        if frame % 100 == 0:
            print("At frame {f}".format(f=frame))

    # Print progress
    print("Finished unpacking subject '{s}', action '{a}', camera '{c}':".format(s=subject, a=action, c=camera))


# Iterate through all of the files (adding all of the videos to be unpacked by a threadpool)
#subjects = [1, 5, 6, 7, 8, 9, 11]
subjects = [11]
with ThreadPoolExecutor(max_workers=num_workers) as pool:
    for subject in subjects:
        subject_dir = os.path.join(base_dir, "S{subject}/Videos/".format(subject=subject))
        for filename in os.listdir(subject_dir):
            # Get the action and camera from filename. File format: action_name.camera_name.mp4
            action, camera, _ = filename.split(".")

            # Skip the video that collects all of them together
            if action == "_ALL" or action == "_ALL 1":
                continue

            # Compute and make the video specific output dir
            vid_output_dir = os.path.join(output_dir, "{s}/{a}/{c}".format(s=subject, a=action, c=camera))
            if not os.path.isdir(vid_output_dir):
                os.makedirs(vid_output_dir)

            # Add job to pool
            full_filename = os.path.join(subject_dir, filename)
            pool.submit(process_video, subject, action, camera, full_filename, vid_output_dir)


