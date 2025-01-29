import cv2
import imutils

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Asking the user to select an image or video file
print("Please select an image or video file for pedestrian detection:")
print("1. Image")
print("2. Video")
choice = int(input("Enter your choice (1 or 2): "))

if choice == 1:
    # Reading the Image file
    image_path = input("Enter the path to the image file: ")
    image = cv2.imread(image_path)

    # Resizing the Image
    image = imutils.resize(image, width=min(1000, image.shape[1]))

    # Detecting all the regions in the Image that has pedestrians inside it
    (regions, _) = hog.detectMultiScale(image,
                                        winStride=(4, 4),
                                        padding=(8, 8),
                                        scale=1.05)

    # Drawing the regions in the Image
    head_count = 0
    for (x, y, w, h) in regions:
        if h >= 80:  # filter out detections with height less than 80 pixels
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 2)
            head_count += 1

    # Showing the output Image
    cv2.putText(image, f"Head count: {head_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif choice == 2:
    # Reading the Video file
    video_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_path)

    # Get the original frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Calculate delay in milliseconds

    while cap.isOpened():
        # Reading the video stream
        ret, image = cap.read()
        if ret:
            image = imutils.resize(image, width=min(800, image.shape[1]))

            # Detecting all the regions in the Image that has pedestrians inside it
            (regions, _) = hog.detectMultiScale(image,
                                                winStride=(4, 4),
                                                padding=(8, 8),
                                                scale=1.05)

            # Drawing the regions in the Image
            head_count = 0
            for (x, y, w, h) in regions:
                if h >= 80:  # filter out detections with height less than 80 pixels
                    cv2.rectangle(image, (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255), 2)
                    head_count += 1

            # Showing the output Image
            cv2.putText(image, f"Head count: {head_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Video", image)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Please try again.")
