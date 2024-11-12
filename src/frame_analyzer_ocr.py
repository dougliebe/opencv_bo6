# frame_analyzer.py
import pytesseract
import cv2
import numpy as np
from ocr_processor import preprocess_frame  # Import preprocess_frame function
# import template_creator
# import text_extractor  # Import the new module

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

def get_pytesseract_text(frame):
    """
    Process the frame and return the OCR result.
    
    Parameters:
    - frame (numpy.ndarray): Frame to be processed.
    
    Returns:
    - ocr_result (str): Extracted text from the frame.
    """
    
    # Configure Tesseract to use the custom language `eng`
    custom_config = (
        r'-l eng '  # Use "eng" language
        r'--oem 3 --psm 7 '  # OCR Engine Mode and Page Segmentation Mode
        r'-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[ ]"'  # Whitelist characters
    )
    
    ocr_result = pytesseract.image_to_string(frame, config=custom_config)
    return ocr_result

if __name__ == "__main__":
    import cv2
    import video_reader  # Import capture_frame function
    import ocr_processor  # Import preprocess_frame function

    image_path = "./images/OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6 31-55 screenshot.png"  # Replace with your actual video file path
    image = video_reader.capture_image_frame(image_path)  # Change frame_number to view different frames
    
    if image is not None:
        top_left = (18, 374)
        bottom_right = (250, 395)
        cropped_image = crop_frame(image, top_left, bottom_right)
        ## process image
        processed_frame = ocr_processor.preprocess_frame(cropped_image)
        
        ocr_result = get_pytesseract_text(cropped_image)
        print(f"Extracted text: {ocr_result}")
        
        cv2.imshow("Cropped Image", processed_frame)
        cv2.waitKey(0)  # Press any key to close the window
        cv2.destroyAllWindows()
    else:
        print("Failed to capture or preprocess frame.")
