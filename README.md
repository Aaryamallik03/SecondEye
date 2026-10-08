# SecondEye



**An AI-powered computer vision assistant for real-time face recognition, spatial awareness, and text reading.**



## Overview



SecondEye is a local computer-vision prototype designed to provide spoken environmental awareness through a webcam.



The system combines face detection, face recognition, multi-face tracking, spatial positioning, optical character recognition (OCR), and voice feedback into a single application.



It is currently a research and development prototype and is **not intended for safety-critical or medical use**.



## Features



| Feature              | Technology                        |

| -------------------- | --------------------------------- |

| Face Detection       | OpenCV Haar Cascade               |

| Face Recognition     | DeepFace + FaceNet512             |

| Multi-Face Tracking  | Custom tracking logic             |

| Spatial Awareness    | Left / Center / Right positioning |

| Voice Feedback       | Windows Speech Synthesis          |

| OCR                  | EasyOCR                           |

| Background OCR       | Worker thread + queue             |

| Face Database        | Local embeddings                  |

| Real-Time Processing | OpenCV webcam pipeline            |



## System Architecture



```text

&#x20;                   Webcam

&#x20;                      │

&#x20;                      ▼

&#x20;             Face Detection

&#x20;            OpenCV Haar Cascade

&#x20;                      │

&#x20;                      ▼

&#x20;             Multi-Face Tracking

&#x20;                      │

&#x20;             ┌────────┴────────┐

&#x20;             ▼                 ▼

&#x20;     Face Recognition     Spatial Position

&#x20;       DeepFace +          Left / Center /

&#x20;       FaceNet512              Right

&#x20;             │                 │

&#x20;             └────────┬────────┘

&#x20;                      ▼

&#x20;                Voice Feedback

&#x20;                 SpeechManager





&#x20;                 T key pressed

&#x20;                      │

&#x20;                      ▼

&#x20;                   EasyOCR

&#x20;                      │

&#x20;                      ▼

&#x20;                Spoken Text

```



## How It Works



### 1. Face Detection



OpenCV's Haar Cascade detector identifies faces from the webcam stream.



### 2. Face Tracking



Detected faces are tracked across frames so that the system can maintain a stable identity and position for multiple people.



### 3. Face Recognition



Face crops are processed using DeepFace with the FaceNet512 model. The resulting embeddings are compared against the locally stored face database.



### 4. Spatial Awareness



The position of each tracked face is classified as:



* Left

* Center

* Right



Position changes are stabilized before being announced to reduce unnecessary voice feedback.



### 5. Voice Feedback



`SpeechManager` handles spoken notifications in the background.



The system can announce:



* Recognized people

* New people

* Changes in spatial position

* Text detected through OCR



### 6. OCR



Pressing **T** triggers text recognition using EasyOCR. OCR processing is handled separately so that text recognition does not unnecessarily block the main camera loop.



## Tech Stack



* Python

* OpenCV

* DeepFace

* FaceNet512

* EasyOCR

* NumPy

* Ultralytics YOLO

* Windows PowerShell System.Speech



## Project Structure



```text

SecondEye/

│

├── recognize.py

├── register.py

├── create\_embeddings.py

├── speech\_manager.py

├── object\_detection.py

├── app.py

│

├── tests/

│   ├── test\_database.py

│   ├── test\_multiple\_faces.py

│   ├── test\_recognition.py

│   └── test\_voice.py

│

├── data/

│   ├── faces/

│   └── database/

│

├── object\_classifier/

│

├── requirements.txt

├── .gitignore

└── README.md

```



### Main Files



**`recognize.py`**

Main application containing face detection, recognition, tracking, spatial awareness, OCR, and voice feedback.



**`register.py`**

Used to register a person's face images.



**`create\_embeddings.py`**

Creates the face embedding database used for recognition.



**`speech\_manager.py`**

Manages background speech generation and voice announcements.



**`object\_detection.py`**

Standalone YOLO-based object detection experiment.



**`app.py`**

Basic OpenCV Haar-cascade face detection demo.



**`tests/`**

Contains tests for database handling, recognition, multiple-face tracking, and voice functionality.



## Installation



Clone the repository and enter the project directory:



```bash

git clone <repository>

cd SecondEye

```



Create a virtual environment:



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



Install the dependencies:



```powershell

pip install -r requirements.txt

```



Some machine-learning dependencies used by DeepFace and EasyOCR may take some time to install.



## Registering a Face



Register a person using:



```powershell

python register.py

```



Then generate the face embeddings:



```powershell

python create\_embeddings.py

```



The generated local face data is stored under the `data/` directory.



## Running SecondEye



Start the main application with:



```powershell

python recognize.py

```



A webcam is required.



### Controls



| Key | Action               |

| --- | -------------------- |

| `Q` | Quit the application |

| `T` | Trigger OCR          |



## Voice Feedback



SecondEye uses a dedicated `SpeechManager` to process voice announcements.



This separates speech generation from the main computer-vision loop and helps keep real-time processing responsive.



Voice feedback can include:



* Person recognition

* New person detection

* Spatial position changes

* OCR results



## Privacy



SecondEye uses locally stored face images and face embeddings for recognition.



Biometric data is intentionally excluded from the Git repository through `.gitignore`.



Users should treat registered face images and embeddings as sensitive personal data and avoid sharing them publicly.



## Limitations



SecondEye is an experimental computer-vision project and has several limitations:



* Recognition accuracy can vary with lighting, camera quality, pose, and distance.

* Haar Cascade detection can miss faces or produce false detections.

* Face recognition can produce false matches or fail to recognize known people.

* OCR performance depends on image quality, text size, orientation, and lighting.

* Voice synthesis currently relies on Windows speech functionality.

* Object detection through YOLO is currently a separate experiment and is not integrated into the main recognition pipeline.

* The system has not been validated for safety-critical, medical, or accessibility-critical applications.



## Future Improvements



Planned improvements include:



* Integrating object detection into the main pipeline

* Supporting more robust face detection models

* Improving multi-person tracking

* Improving recognition stability

* Adding configurable voice settings

* Improving OCR accuracy and text filtering

* Adding cross-platform speech support

* Adding performance benchmarking

* Expanding automated tests

* Moving configuration values into a dedicated configuration system



## Project Goal



The goal of SecondEye is to explore how computer vision, face recognition, spatial awareness, OCR, and speech synthesis can be combined into a real-time assistive computer-vision system.



The project is primarily intended for learning, experimentation, and further development.



## Author



**Aarya Mallik**



B.Tech Computer Science and Engineering

NIT Meghalaya



GitHub repository: `Aaryamallik03/SecondEye`



## Disclaimer



SecondEye is an educational and research prototype. It should not be relied upon as a replacement for professional accessibility, navigation, medical, or safety systems.




