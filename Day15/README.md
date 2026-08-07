Day 16 - OpenCV Fundamentals & Image Processing Toolkit 🖼️





Part of the MLBench Summer Internship Program.

This module focuses on the fundamentals of OpenCV, exploring how digital images are represented and manipulated through various image processing operations while building an interactive Image Processing Toolkit using Streamlit.

📌 Overview
Task	Image Processing Toolkit using OpenCV
Library	OpenCV (cv2)
Language	Python
Framework	Streamlit
Input	User Uploaded Images
Output	Processed Images
Deployment	Streamlit Image Processing Toolkit
🧠 Concepts Covered
What is OpenCV?

OpenCV (Open Source Computer Vision Library) is one of the world's most widely used computer vision libraries. It provides hundreds of optimized functions for image processing, video analysis, feature extraction, machine learning, and real-time computer vision applications.

OpenCV is commonly used in:

Image Processing
Face Recognition
Medical Imaging
Robotics
Autonomous Vehicles
Industrial Inspection
🖼️ BGR vs RGB

Digital images store color information using three channels.

BGR	RGB
Blue → Green → Red	Red → Green → Blue
Default color format used by OpenCV	Standard format used by Pillow and Matplotlib
Optimized for OpenCV processing	Better suited for visualization

Since OpenCV loads images in BGR, converting them to RGB is necessary before displaying them with visualization libraries.

⚫ What are Grayscale Images?

A grayscale image contains only one intensity channel instead of three color channels.

Each pixel stores brightness values ranging from:

0   → Black
255 → White

Grayscale images are widely used because they:

Reduce computational complexity
Simplify image analysis
Improve edge detection
Enhance feature extraction
Reduce storage requirements
📂 OpenCV Practice Programs

The project contains several standalone programs demonstrating OpenCV fundamentals.

1️⃣ OpenCV Fundamentals

File

opencv_fundamentals.py

Features:

Read images
Display image dimensions
Display image channels
Display file size
Convert BGR → RGB
Convert to Grayscale
Save processed images
2️⃣ Basic Image Operations

File

basic_image_operations.py

Implemented operations:

Resize images
Crop images
Rotate images
Flip horizontally
Flip vertically
Save processed outputs
3️⃣ Drawing Shapes

File

drawing_shapes.py

Using OpenCV drawing functions:

Rectangle
Circle
Line
Polygon
Custom Text (Name & Date)

All generated images are automatically saved inside the drawings folder.

🛠️ Mini Project — Image Processing Toolkit

File

image_processing_toolkit.py

This project combines all OpenCV operations into a reusable menu-driven toolkit.

Users can:

Load an image
Convert to grayscale
Resize image
Rotate image
Flip image
Crop image
Draw shapes
Add custom text
Save processed image
🌐 Streamlit Image Processing Toolkit

File

streamlit_app.py

Live Application

https://mlb-internship-iofdnxjvjdatkpvbmofjjd.streamlit.app/

The project includes an interactive Streamlit application where users can upload an image and perform various OpenCV operations directly in the browser.

🚀 Streamlit Features
Image Upload

Supported formats:

JPG
JPEG
PNG
BMP
Image Processing Operations

Users can perform:

Convert to Grayscale
Resize Image
Rotate Image
Flip Horizontally
Flip Vertically
Crop Image
Drawing Tools

Interactive drawing features include:

Rectangle
Circle
Line
Polygon
Custom Text
Color Picker
Thickness Slider
Undo Last Drawing
Export

Users can:

Preview processed image
Download processed image
📊 OpenCV Functions Used

The following OpenCV functions were implemented throughout the project:

cv2.imread()
cv2.imwrite()
cv2.cvtColor()
cv2.resize()
cv2.rotate()
cv2.flip()
cv2.rectangle()
cv2.circle()
cv2.line()
cv2.polylines()
cv2.putText()
📚 Observations

During experimentation:

OpenCV reads images using the BGR color format.
RGB conversion is required for correct visualization outside OpenCV.
Grayscale images simplify processing while preserving essential structural information.
Image resizing changes dimensions without affecting the original image.
Cropping allows extraction of regions of interest.
Rotation and flipping efficiently change image orientation.
Drawing functions make it easy to annotate images for visualization.
Interactive controls significantly improve usability for image editing.
🗂️ Project Structure
Day16/
│
├── opencv_fundamentals.py
├── basic_image_operations.py
├── drawing_shapes.py
├── image_processing_toolkit.py
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── input_images/
│
├── output_images/
│
└── screen_recording.mp4
⚙️ Installation & Usage
Install Dependencies
pip install -r requirements.txt
Run OpenCV Fundamentals
python opencv_fundamentals.py
Run Basic Image Operations
python basic_image_operations.py
Run Drawing Shapes
python drawing_shapes.py
Run Image Processing Toolkit
python image_processing_toolkit.py
Launch Streamlit Application
streamlit run streamlit_app.py
🌍 Deployment
GitHub Repository

https://github.com/HadeedJalani/MLB-Internship/tree/main/Day16

Streamlit Application

https://mlb-internship-iofdnxjvjdatkpvbmofjjd.streamlit.app/

📚 Learning Outcomes

Through this project, I learned:

Fundamentals of OpenCV
Digital image representation
Difference between BGR and RGB
Grayscale image processing
Image resizing, cropping, rotation, and flipping
Drawing geometric shapes using OpenCV
Building reusable image processing functions
Developing an interactive Streamlit application
Deploying OpenCV applications on Streamlit Cloud
👨‍💻 Author

Hadeed Jalani

BS Computer Science

MLBench Summer Internship
