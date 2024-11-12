import cv2
import numpy as np
import template_creator
import ocr_processor


def extract_text_from_roi(frame, possible_texts, template_dir="templates"):
    """
    Extract text from a specified region of interest (ROI) in the frame using template matching.
    
    Parameters:
    - frame (numpy.ndarray): Preprocessed frame.
    - possible_texts (list): List of possible texts to match.
    - template_dir (str): Directory containing the template images.
    
    Returns:
    - matches (list): List of tuples containing the matched texts, their confidence values, and x coordinates.
    """
    matches = []
    for text in possible_texts:
        template_path = template_creator.create_template(text, output_dir=template_dir)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        ## print size of template and frame
        print(template.shape)
        print(frame.shape)
        # template = cv2.resize(template, (frame.shape[1], frame.shape[0]))
        edges_frame = cv2.Canny(frame, 50, 150)
        edges_template = cv2.Canny(template, 50, 150)
        contours_frame, _ = cv2.findContours(edges_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_template, _ = cv2.findContours(edges_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_frame and contours_template:
            similarity = cv2.matchShapes(contours_frame[0], contours_template[0], cv2.CONTOURS_MATCH_I1, 0.0)
            matches.append((text, similarity, 0))  # x-coordinate is not relevant for contour matching
    
    return matches

if __name__ == "__main__":
    # Test the extract_text_from_roi function
    image_path = "d:/Downloads/OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6 31-55 screenshot.png"  # Replace with your actual video file path
    image = cv2.imread(image_path)
    top_left = (18, 374)
    bottom_right = (250, 393)
    cropped_image = ocr_processor.crop_frame(image, top_left, bottom_right)
    
    preprocess_frame = ocr_processor.preprocess_frame(cropped_image)
    possible_texts = [
        "[100T] Ghosty ", "[100T] Envoy ", "[100T] HyDra ", "[100T] Scrap ",
        "[OG] DashySZN ", "[OG] Shotzzy ", "[OG] Kylo Ken ", "[OG] Pred "
    ]
    top_matches = extract_text_from_roi(preprocess_frame, possible_texts)
    for match in top_matches:
        print(f"Text: {match[0]}, Contour Similarity: {match[1]}, x-coordinate: {match[2]}")
    
    ## show images
    cv2.imshow("Cropped Image", preprocess_frame)
    cv2.waitKey(0)  # Press any key to close the window
    cv2.destroyAllWindows()