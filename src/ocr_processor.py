# ocr_processor.py
import cv2


def crop_frame(frame, top_left, bottom_right):
    """
    Crop the frame to the specified region.
    
    Parameters:
    - frame (numpy.ndarray): Original frame.
    - top_left (tuple): Coordinates of the top-left corner (x, y).
    - bottom_right (tuple): Coordinates of the bottom-right corner (x, y).
    
    Returns:
    - cropped_frame (numpy.ndarray): Cropped frame.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame

def preprocess_frame(frame):
    """
    Preprocess the frame to improve OCR accuracy.
    
    Parameters:
    - frame (numpy.ndarray): Original frame captured from the video.
    
    Returns:
    - processed_frame (numpy.ndarray): Preprocessed frame ready for OCR.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    
    # # Upscale the image
    # upscale_factor = 3
    # upscaled = cv2.resize(clahe_applied, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Apply thresholding
    _, thresh = cv2.threshold(clahe_applied, 125, 255, cv2.THRESH_BINARY)
    
    # ## print size of image
    # print(f"Image size: {thresh.shape}")
    
    return thresh

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")

if __name__ == "__main__":
    import video_reader  # Import capture_frame function

    image_path = "images\OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6 31-55 screenshot.png"  # Replace with your actual video file path
    image = video_reader.capture_image_frame(image_path)  # Change frame_number to view different frames
    
    if image is not None:
        processed_frame = preprocess_frame(image)
        cv2.imshow("Preprocessed Frame", processed_frame)
        cv2.setMouseCallback("Preprocessed Frame", on_mouse_click)
        cv2.waitKey(0)  # Press any key to close the window
        cv2.destroyAllWindows()
    else:
        print("Failed to capture or preprocess frame.")
