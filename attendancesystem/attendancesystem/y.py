import cv2
import os

def newstudent(nameID):
    # Define the paths to the model files
    prototxt_path = r"C:\Users\Pranshu Saini\Downloads\deploy.prototxt"
    model_path = r"C:\Users\Pranshu Saini\Downloads\res10_300x300_ssd_iter_140000.caffemodel"

    # Check if the files exist
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model files not found. Check the paths.")

    # Load the DNN model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    video = cv2.VideoCapture(0)

    # Get the student's name
    path = os.path.join("C:/Users/Pranshu Saini/Desktop/images", nameID)

    # Check if the directory already exists
    if os.path.exists(path):
        print("Name Already Taken. Please choose another name.")
        nameID = str(input("Enter Your Name Again: ")).lower()
        path = os.path.join("C:/Users/Pranshu Saini/Desktop/images", nameID)

    # Create the directory
    os.makedirs(path)

    count = 0  # Initialize image counter

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error accessing the webcam.")
            break

        # Prepare the frame for the DNN model
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:  # Save only confident detections
                count += 1
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")

                # Ensure the box doesn't exceed image dimensions
                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)

                # Save the detected face as an image
                face = frame[y:y1, x:x1]
                name = os.path.join(path, f"{count}.jpg")
                print(f"Creating image... {name}")
                cv2.imwrite(name, face)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Show the frame with the detected face
        cv2.imshow("WindowFrame", frame)

        # Stop capturing if 100 images are saved
        if count >= 500:
            print("Captured 500 images. Stopping...")
            break

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    # Release the video and destroy all OpenCV windows
    video.release()
    cv2.destroyAllWindows()

    return "Images taken successfully!"
