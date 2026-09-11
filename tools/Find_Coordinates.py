import cv2

coordinates = []

def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert back to original image coordinates
        orig_x = int(x * param['scale_x'])
        orig_y = int(y * param['scale_y'])
        print(f"Clicked at: ({orig_x}, {orig_y})")
        coordinates.append((orig_x, orig_y))

# Load image
image = cv2.imread(r"frames/multi.png")
if image is None:
    print("Failed to load image. Check the file path!")
    exit()


##Scale the image to fit on your screen

# Original dimensions
orig_height, orig_width = image.shape[:2]
# Set max dimensions (screen size or comfortable window size)
max_width, max_height = 1400, 900 
# Only scale down if image is bigger than max dimensions
scale = min(max_width / orig_width, max_height / orig_height, 1)  # never upscale
new_width = int(orig_width * scale)
new_height = int(orig_height * scale)
resized_image = cv2.resize(image, (new_width, new_height))

# Show window
cv2.imshow('Image', resized_image)

# Set mouse callback
cv2.setMouseCallback('Image', get_coordinates, {'scale_x': 1/scale, 'scale_y': 1/scale})

print("Click on each cone to record coordinates. Press 'q' to quit.")

while True:
    cv2.imshow('Image', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("All recorded coordinates (relative to original image):")
print(coordinates)
print("Image height and width:",image.shape[:2])
