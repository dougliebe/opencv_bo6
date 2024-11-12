import cv2
import numpy as np

# Create a blank image
image = np.zeros((400, 800, 3), dtype="uint8")

# List of OpenCV fonts
fonts = [
    ("FONT_HERSHEY_SIMPLEX", cv2.FONT_HERSHEY_SIMPLEX),
    ("FONT_HERSHEY_PLAIN", cv2.FONT_HERSHEY_PLAIN),
    ("FONT_HERSHEY_DUPLEX", cv2.FONT_HERSHEY_DUPLEX),
    ("FONT_HERSHEY_COMPLEX", cv2.FONT_HERSHEY_COMPLEX),
    ("FONT_HERSHEY_TRIPLEX", cv2.FONT_HERSHEY_TRIPLEX),
    ("FONT_HERSHEY_COMPLEX_SMALL", cv2.FONT_HERSHEY_COMPLEX_SMALL),
    ("FONT_HERSHEY_SCRIPT_SIMPLEX", cv2.FONT_HERSHEY_SCRIPT_SIMPLEX),
    ("FONT_HERSHEY_SCRIPT_COMPLEX", cv2.FONT_HERSHEY_SCRIPT_COMPLEX),
    ("FONT_ITALIC (Example with SIMPLEX)", cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC),
]

# Display each font
y = 20
for font_name, font in fonts:
    cv2.putText(image, f"{font_name}", (10, y), font, 1, (255, 255, 255), 2)
    y += 40

# Show the image with font examples
cv2.imshow("OpenCV Fonts", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
