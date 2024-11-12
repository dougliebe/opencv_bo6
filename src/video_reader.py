import cv2

def capture_frame(video_path, frame_number=0):
    """
    Capture a specific frame from a video file.
    
    Parameters:
    - video_path (str): Path to the video file.
    - frame_number (int): Frame number to capture (default is 0 for the first frame).
    
    Returns:
    - frame (numpy.ndarray): Captured frame as an image array, or None if unsuccessful.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Failed to capture frame.")
        return None

def capture_image_frame(image_path):
    """
    Capture an image from a PNG file.
    
    Parameters:
    - image_path (str): Path to the PNG file.
    
    Returns:
    - image (numpy.ndarray): Captured image as an image array, or None if unsuccessful.
    """
    image = cv2.imread(image_path)
    if image is not None:
        return image
    else:
        print(f"Error: Could not open image file {image_path}")
        return None

# Test block to capture and display a single frame
if __name__ == "__main__":
    image_path = ".\images\OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6 31-55 screenshot.png"  # Replace with your actual video file path
    image = capture_image_frame(image_path)  # Change frame_number to view different frames
    
    if image is not None:
        print("Displaying captured image...")
        cv2.imshow("Captured Image", image)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
    else:
        print("No image captured.")
