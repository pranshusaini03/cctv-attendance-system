# Attendance System (Django + TensorFlow + OpenCV)

A web-based face recognition attendance system built with Django for the web interface and TensorFlow/Keras + OpenCV for model training and real-time recognition. Attendance records are stored in an Excel file using OpenPyXL.

### Features
- Real-time face detection using either OpenCV DNN (Caffe model) or MTCNN
- Face recognition using a TensorFlow/Keras CNN (VGG16-based)
- Attendance logging to `attendance.xlsx`
- Simple Django UI to train model, take attendance, register/view

### Project structure
- `attendancesystem/`: Django project with views and ML scripts
- `templates/`: HTML templates for pages
- `face_recognition_model.h5`: Trained Keras model (provided or generated)
- `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`: OpenCV DNN face detector files
- `attendance.xlsx`: Excel file with attendance logs

### Prerequisites
- Python 3.10–3.12 (ensure a version compatible with your TensorFlow build)
- pip
- A working webcam or IP camera stream

### Setup
1. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows PowerShell
. .venv\\Scripts\\Activate.ps1
# or cmd
.venv\\Scripts\\activate.bat
```

2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Place/verify model and detector files in project root
- `face_recognition_model.h5`
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

4. Verify/adjust hardcoded paths in code
- Image dataset path used for training/labels: update `images_path` in `attendancesystem/dnn.py` and `attendancesystem/mtcn.py`
- Excel file path if you want a specific location
- Optional IP camera address in `take_attendance` / `take_attendancemtcnn`

### Run the web app
```bash
python manage.py migrate
python manage.py runserver
```
Open `http://127.0.0.1:8000/` in your browser.

### Training the model
The function `train_model` in `attendancesystem/model.py` trains a VGG16-based classifier using images organized as:
```
images/
  person_a/
    img1.jpg
    img2.jpg
  person_b/
    img1.jpg
    ...
```
You can trigger training via the UI page that calls `train_model`, or programmatically:
```bash
python -c "from attendancesystem.model import train_model; train_model(r'PATH_TO_IMAGES')"
```
This saves `face_recognition_model.h5` in the project root by default.

### Taking attendance
The app supports two detection backends:
- OpenCV DNN (Caffe) via `attendancesystem/dnn.py`
- MTCNN via `attendancesystem/mtcn.py`

When running, a window shows detections. Click the on-screen button labeled "Attendance" to append the recognized name and confidence to `attendance.xlsx`. Press `q` to quit.

### Notes
- TensorFlow and OpenCV wheels can be large; installation may take time.
- Ensure your TensorFlow version supports your Python version and CPU/GPU.
- If you get errors loading the Caffe model, verify `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` paths.
- For Windows, if you face issues with camera access, make sure camera permissions are enabled.

### License
This project is provided as-is for educational purposes. 
