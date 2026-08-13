# 🎥 Day 20  Video Processing with OpenCV

### MLB Summer Internship — Computer Vision Track

> **A real-time and recorded-video processing application built with Python, OpenCV, and Streamlit.**

---

## 📌 Project Overview

Day 20 focuses on **Video Processing with OpenCV**.

A video can be understood as a sequence of individual image frames. By processing these frames one at a time, computer vision systems can perform tasks such as grayscale conversion, noise reduction, edge detection, object analysis, surveillance, traffic monitoring, and video enhancement.

For this project, I developed an interactive **Video Processing Tool** capable of processing both:

- 🎬 Uploaded / recorded videos
- 📷 Live webcam streams

The application provides multiple processing modes and allows users to observe how each technique changes the video frames.

---

# 🎯 Objectives

The main objectives of this project were to:

- Understand how OpenCV reads video files.
- Understand video frames and frame-by-frame processing.
- Extract important video properties.
- Understand FPS and video duration.
- Convert video frames to grayscale.
- Apply Gaussian Blur.
- Apply Canny Edge Detection.
- Save processed videos.
- Process webcam frames in real time.
- Build an interactive Streamlit interface.
- Compare original and processed video output.
- Create a reusable video-processing pipeline.

---

# 🧠 Concepts Covered

## 1. Video as a Sequence of Frames

A video is essentially a sequence of images displayed rapidly over time.

For example:

```text
Frame 1 → Frame 2 → Frame 3 → Frame 4 → ... → Frame N
OpenCV allows us to read these frames individually and apply image-processing operations to each frame.

This means many image-processing techniques learned previously can also be applied to videos.

🎞️ Video Properties

The application extracts important information from uploaded videos, including:

Property	Description
FPS	Frames processed/displayed per second
Width	Video frame width in pixels
Height	Video frame height in pixels
Total Frames	Number of frames in the video
Duration	Approximate video duration
FPS — Frames Per Second

FPS represents the number of frames displayed or processed every second.

For example:

30 FPS

means the video contains approximately 30 frames for every second of playback.

Higher FPS generally produces smoother motion, while lower FPS can make motion appear less smooth.

⚫ Grayscale Conversion

The first processing technique converts each color frame into a grayscale image.

Instead of storing three color channels:

Blue
Green
Red

the grayscale image contains a single intensity channel.

Why grayscale?

Grayscale simplifies image processing because many computer vision algorithms only need intensity information rather than full color information.

It is especially useful before:

Edge detection
Thresholding
Contour detection
Shape detection
OCR
Image segmentation
🌫️ Gaussian Blur

Gaussian Blur is applied to reduce noise and smooth the image.

The process applies a Gaussian kernel over the image.

In this project, the blur kernel can be adjusted interactively.

Example:

Original Frame
      ↓
Grayscale
      ↓
Gaussian Blur
      ↓
Edge Detection
Why use Gaussian Blur before Canny?

Real-world video frames often contain small variations and noise.

If edge detection is applied directly, these small variations may be incorrectly detected as edges.

Gaussian Blur helps smooth these variations before edge detection.

✨ Canny Edge Detection

Canny Edge Detection identifies strong changes in image intensity.

The general pipeline used in this project is:

Video Frame
     ↓
Grayscale
     ↓
Gaussian Blur
     ↓
Canny Edge Detection
     ↓
Edge Frame

Canny is useful for detecting:

Object boundaries
Shapes
Structural edges
Document boundaries
Road boundaries
Object outlines
🎛️ Canny Thresholds

The application provides two adjustable Canny threshold values:

Lower Threshold

Controls the minimum intensity difference required for edge detection.

Upper Threshold

Controls the stronger edge threshold.

The user can experiment with these values to find an appropriate balance.

For example:

Lower Threshold = 50
Upper Threshold = 150

Different videos may require different threshold values depending on:

Lighting
Noise
Contrast
Object texture
Camera quality
Background complexity
📷 Live Webcam Processing

One of the main features of this project is real-time webcam processing.

The application uses:

Streamlit
      ↓
streamlit-webrtc
      ↓
Browser Camera
      ↓
Live Video Frames
      ↓
OpenCV Processing
      ↓
Processed Frames

The webcam stream can be processed using different modes:

Original
Grayscale
Gaussian Blur
Canny Edge Detection

The processing occurs frame-by-frame in real time.

🌐 Why streamlit-webrtc?

For a deployed Streamlit application, using:

cv2.VideoCapture(0)

is not sufficient for accessing the visitor's webcam.

That approach attempts to access the camera of the machine running the Python application.

Instead, this project uses:

streamlit-webrtc

to establish communication between the browser's webcam and the Streamlit application.

This allows the application to process webcam frames from the user's browser.

🎬 Recorded Video Processing

Users can upload videos in formats such as:

MP4
AVI
MOV
MKV
WEBM

The application then:

Loads the uploaded video.
Reads the video properties.
Extracts individual frames.
Applies the selected processing technique.
Writes the processed frames into a new video.
Displays the processed output.
Provides the processed video for download.
🔍 Original vs Processed Comparison

The application provides a side-by-side comparison:

┌─────────────────────┬─────────────────────┐
│                     │                     │
│   Original Video    │   Processed Video   │
│                     │                     │
│                     │                     │
└─────────────────────┴─────────────────────┘

This makes it easier to visually understand the effect of each processing technique.

For example:

Original

A normal RGB/color video frame.

Grayscale

The same frame represented using intensity values.

Gaussian Blur

The frame becomes smoother and small details/noise are reduced.

Canny

Strong edges and boundaries are highlighted.

🖥️ Application Features
🎥 Recorded Video Mode

The recorded-video section provides:

Video upload
Supported video formats
FPS information
Resolution information
Total frame count
Duration
Processing mode selection
Adjustable Gaussian Blur
Adjustable Canny thresholds
Processing progress
Processing statistics
Original video preview
Processed video preview
Side-by-side comparison
Processed video download
📷 Webcam Mode

The webcam section provides:

Browser camera access
Live video streaming
Real-time frame processing
Original mode
Grayscale mode
Gaussian Blur
Canny Edge Detection
Adjustable blur kernel
Adjustable Canny thresholds
📊 Processing Statistics

After processing a recorded video, the application displays useful statistics such as:

Frames processed
Processing time
Video FPS
Video resolution

Example:

Frames Processed : 900
Processing Time  : 8.42 sec
FPS              : 30
Resolution       : 1280 × 720

These statistics provide a basic understanding of the processing workload and video characteristics.

🗂️ Project Structure

The Day 20 project is organized as follows:

Day20/
│
├── streamlit_app.py
│
├── video_operations.py
│
├── requirements.txt
│
├── README.md
│
├── input_videos/
│   └── sample videos
│
└── output_videos/
    └── processed videos

Input and output media files can be kept locally and do not need to be committed to the GitHub repository.

📄 File Description
streamlit_app.py

Main Streamlit application.

Responsible for:

User interface
Video upload
Processing controls
Video comparison
Statistics
Webcam interface
Download functionality
video_operations.py

Contains the core OpenCV video-processing logic.

Responsible for:

Opening videos
Reading frames
Extracting video properties
Processing frames
Applying grayscale conversion
Applying Gaussian Blur
Applying Canny
Writing processed videos
Tracking processing progress
requirements.txt

Contains the Python dependencies required to run the application.

Main dependencies include:

streamlit
opencv-python-headless
numpy
av
streamlit-webrtc
⚙️ Installation
1. Clone the Repository
git clone https://github.com/HadeedJalani/MLB-Internship.git

Navigate into the project:

cd MLB-Internship/Day20
2. Create a Virtual Environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Application

Start the Streamlit application:

streamlit run streamlit_app.py

The application will open in the browser.

🧪 Testing

The application was tested using different processing modes and video inputs.

The main test cases include:

Test 1 — Original

Expected result:

Video remains unchanged.
Test 2 — Grayscale

Expected result:

Color information is removed
and frames become grayscale.
Test 3 — Gaussian Blur

Expected result:

Frames become smoother and
high-frequency noise is reduced.
Test 4 — Canny Edge Detection

Expected result:

Strong object boundaries and
intensity transitions become visible.
🏆 Challenge Task

The Day 20 challenge requires processing three different videos.

For each video:

Original Video
      ↓
Processing
      ↓
Processed Video
      ↓
Comparison

The objective is to observe how different video content affects the results of:

Grayscale conversion
Gaussian Blur
Canny Edge Detection
Recommended Test Videos

The application can be tested with videos containing:

People
Vehicles
Outdoor scenes
Indoor scenes
Objects with strong edges
Objects with low contrast
🔎 Observations

During testing, several important observations can be made.

Grayscale

Grayscale significantly reduces the amount of visual information while preserving intensity structure.

This makes subsequent computer vision operations easier.

Gaussian Blur

Gaussian Blur is particularly useful when the input contains noise.

However, excessive blur can remove useful details and make edges less distinct.

Therefore, the kernel size needs to be selected carefully.

Canny

Canny provides strong edge representations, but the result depends heavily on the threshold values.

Low thresholds may produce:

Too many edges
+ noise
+ unwanted details

Very high thresholds may produce:

Too few edges
+ missing object boundaries

Therefore, threshold tuning is important.

⚠️ Challenges

Some of the challenges encountered during development included:

1. Different Video Formats

Different video formats and codecs can behave differently when processed with OpenCV.

2. Video Compatibility

A generated video must use a codec and container format compatible with the browser.

This is especially important when displaying processed videos through Streamlit.

3. Processing Speed

High-resolution videos contain a large number of pixels per frame.

Processing every frame can therefore take significant computational time.

4. Canny Threshold Selection

There is no single threshold pair that works perfectly for every video.

Lighting, contrast, texture, and camera conditions affect edge detection.

5. Webcam Access

Accessing a user's webcam from a deployed Streamlit application requires browser-based communication.

For this reason, streamlit-webrtc was used instead of relying only on:

cv2.VideoCapture(0)
💡 Real-World Applications

Video processing is an important foundation for many Computer Vision systems.

Applications include:

🚦 Traffic Monitoring

Detecting vehicles and road boundaries from traffic camera footage.

🛡️ Surveillance

Processing security camera feeds for motion and object analysis.

🚗 Autonomous Vehicles

Analyzing frames from cameras to identify:

Roads
Vehicles
Pedestrians
Lane boundaries
Obstacles
🏭 Industrial Inspection

Analyzing production-line video to identify defects.

🏥 Medical Imaging

Processing medical video streams for analysis and visualization.

📄 Document Processing

Video frames can be processed to detect documents before OCR.

🤖 Object Detection

Edge detection and frame processing are foundational concepts for more advanced object detection pipelines.

🔬 Technologies Used
Technology	Purpose
Python	Programming language
OpenCV	Computer Vision and video processing
NumPy	Numerical image operations
Streamlit	Interactive web application
Streamlit-WebRTC	Browser webcam streaming
PyAV	Video frame handling
🚀 Future Improvements

Possible future improvements include:

Real-time FPS monitoring
Video playback controls
Adjustable processing parameters during playback
Motion detection
Object detection
Background subtraction
Optical flow
Face detection
Object tracking
YOLO integration
Real-time video analytics
Video frame export
Processing history
Multiple processing pipelines
Advanced codec support
📚 Key Learning Outcomes

After completing this project, I gained practical understanding of:

How videos are represented as frames.
How OpenCV reads video streams.
How to extract video properties.
How FPS affects video playback.
How to process frames individually.
How grayscale conversion works.
How Gaussian Blur reduces noise.
How Canny detects edges.
How to save processed videos.
How to build real-time webcam processing.
How browser-based webcam streaming works with Streamlit.
How to build an interactive Computer Vision application.
🌐 Live Application

The project can be deployed using Streamlit.

Live App:

https://mlb-internship-txl3no5nh4hivq8unmbdnx.streamlit.app/

🐙 GitHub Repository

The complete internship work is available in the GitHub repository:

MLB Internship Repository

https://github.com/HadeedJalani/MLB-Internship

🎥 Demo

The project demonstration covers:

Uploading a video.
Inspecting video properties.
Selecting a processing mode.
Applying grayscale conversion.
Applying Gaussian Blur.
Applying Canny Edge Detection.
Comparing original and processed videos.
Downloading processed output.
Opening the webcam processing section.
Processing webcam frames in real time.

Conclusion

Day 20 extended the image-processing concepts from previous tasks into the world of video processing.

The key idea is that a video can be treated as a sequence of individual image frames. This makes it possible to reuse many image-processing techniques for real-time and recorded video applications.

The final application combines:

OpenCV
   +
Frame Processing
   +
Grayscale
   +
Gaussian Blur
   +
Canny Edge Detection
   +
Webcam Streaming
   +
Streamlit

into a single interactive Computer Vision tool.

This project provides a foundation for more advanced topics such as object detection, object tracking, motion detection, optical flow, and real-time video analytics.

👨‍💻 Author
Hadeed Jalani

MLB Summer Internship — Computer Vision

Built with Python, OpenCV, NumPy, Streamlit, PyAV, and Streamlit-WebRTC.
