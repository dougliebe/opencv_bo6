import cv2

# Read the image
image = cv2.imread('templates/scoreboard_names/2 hydra.png')

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not load image.")
    exit()

# Define the kernel for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Dilate the image
dilated_image = cv2.dilate(image, kernel, iterations=1)

# Display the dilated image
cv2.imshow('Dilated Image', dilated_image)

# Wait for any key to be pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()