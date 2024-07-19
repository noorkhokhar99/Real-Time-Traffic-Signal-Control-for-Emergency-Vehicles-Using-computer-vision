from flask import Flask, request, render_template, Response
from inference_sdk import InferenceHTTPClient
import cv2
import concurrent.futures

# Hard-coded API key
api_key = "89oDCCc4u1bHAaLGbEJJ"

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = CLIENT.infer(gray_frame, model_id="ambulance-sjpea/18")
    if 'predictions' in result and result['predictions']:
        for prediction in result['predictions']:
            x, y = int(prediction['x']), int(prediction['y'])
            width, height = int(prediction['width']), int(prediction['height'])
            confidence = prediction['confidence']
            label = prediction['class']

            x1, y1 = x - width // 2, y - height // 2
            x2, y2 = x + width // 2, y + height // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    return frame_bytes

def generate_frames(video_path=None, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video/webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with concurrent.futures.ThreadPoolExecutor() as executor:
            frame_bytes = executor.submit(process_frame, frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.result() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    video_path = "uploaded_video.mp4"
    file.save(video_path)
    return render_template('play.html', video_path=video_path)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    use_webcam = request.args.get('use_webcam', False, type=bool)
    return Response(generate_frames(video_path, use_webcam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
