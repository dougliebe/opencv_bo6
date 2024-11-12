# frame_analyzer.py
import pytesseract
import cv2
import numpy as np
import template_creator
import text_extractor  # Import the new module


if __name__ == "__main__":
    import cv2
    import video_reader  # Import capture_frame function
    import ocr_processor  # Import preprocess_frame function

    image_path = "d:/Downloads/OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6 31-55 screenshot.png"  # Replace with your actual video file path
    image = video_reader.capture_image_frame(image_path)  # Change frame_number to view different frames
    
    if image is not None:
        top_left = (14, 248)
        bottom_right = (177, 262)
        cropped_image = crop_frame(image, top_left, bottom_right)
        ## process image
        processed_frame = ocr_processor.preprocess_frame(cropped_image)
        # extracted_text = pytesseract.image_to_string(processed_frame)
        # print(f"Extracted text: {extracted_text}")
        possible_texts = [
            "[100T] Ghosty ", "[100T] Envoy ", "[100T] HyDra ", "[100T] Scrap ",
            "[OG] DashySZN ", "[OG] Shotzzy ", "[OG] Kylo Ken ", "[OG] Pred "
        ]
        top_matches = text_extractor.extract_text_from_roi(processed_frame, possible_texts)
        for match in top_matches:
            print(f"Text: {match[0]}, Confidence: {match[1]}, x-coordinate: {match[2]}")
        ## put a dot on the image to show the x-coordinate, y coord is half the height
        # for match in top_matches:
        #     cv2.circle(cropped_image, (match[2], cropped_image.shape[0]//2), 5, (0, 0, 255), -1)
        cv2.imshow("Cropped Image", processed_frame)
        cv2.waitKey(0)  # Press any key to close the window
        cv2.destroyAllWindows()
    else:
        print("Failed to capture or preprocess frame.")
