import cv2
import numpy as np
import os

def preprocess_template(frame):
    """
    Preprocess the frame to improve OCR accuracy.
    
    Parameters:
    - frame (numpy.ndarray): Original frame captured from the video.
    
    Returns:
    - processed_frame (numpy.ndarray): Preprocessed frame ready for OCR.
    """
    # # Convert to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(frame)
    
    # Upscale the image
    upscale_factor = 2
    upscaled = cv2.resize(clahe_applied, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Apply thresholding
    _, thresh = cv2.threshold(upscaled, 100, 255, cv2.THRESH_BINARY)
    
    # Make the image 2 pixels taller and 1 pixel thinner
    processed_frame = cv2.resize(thresh, (thresh.shape[1] - 1, thresh.shape[0] + 2), interpolation=cv2.INTER_LINEAR)
    
    return processed_frame

def create_template(text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.39, thickness=1, output_dir="templates"):
    """
    Create a template image with the given text and save it as a PNG file.
    
    Parameters:
    - text (str): Text to write on the template.
    - font (int): Font type.
    - font_scale (float): Font scale.
    - thickness (int): Thickness of the text.
    - output_dir (str): Directory to save the template images.
    
    Returns:
    - template_path (str): Path to the saved template image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    template_path = os.path.join(output_dir, f"{text}.png")
    
    if not os.path.exists(template_path):
        words = text.split(' ')
        word_sizes = [cv2.getTextSize(word, font, font_scale, thickness)[0] for word in words]
        total_width = sum(size[0] for size in word_sizes) + (len(words) - 1) + 2
        max_height = max(size[1] for size in word_sizes) + 5
        
        template = np.zeros((max_height, total_width), dtype=np.uint8)
        
        x_offset = 1
        for word, size in zip(words, word_sizes):
            cv2.putText(template, word, (x_offset, size[1] + 1), font, font_scale, (255, 255, 255), thickness)
            x_offset += size[0] + 1
        
        # Apply preprocessing
        template = preprocess_template(template)
        
        # Apply dilation to slightly increase thickness
        kernel = np.ones((2, 2), np.uint8)
        template = cv2.dilate(template, kernel, iterations=1)
        
        cv2.imwrite(template_path, template)
    
    return template_path

if __name__ == "__main__":
    text = "[100T] Ghosty "
    template_path = create_template(text)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if template is not None:
        cv2.imshow("Template", template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to create template.")