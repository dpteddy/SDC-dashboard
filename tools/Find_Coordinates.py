import cv2

# ------------------------------------------------------------
# This script lets a user manually click points on an image
# and records the coordinates of each click.
#
# It was originally used to collect the court corner coordinates
# before they were stored permanently in cour_track_polygons.py.
#
# Today, you do NOT need this script anymore because your
# coordinates already exist in objp[] inside cour_track_polygons.py.
# ------------------------------------------------------------

coordinates = []   # list to store clicked coordinates


def get_coordinates(event, x, y, flags, param):
    """
    Mouse callback function.
    Triggered whenever the user clicks inside the image window.

    Parameters:
        event : type of mouse event (left-click, right-click, etc.)
        x, y  : coordinates of the click in the *resized* image
        param : dictionary containing scale factors to convert
                resized-image coordinates back to original-image coordinates
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert resized-image coordinates back to original-image coordinates
        orig_x = int(x * param['scale_x'])
        orig_y = int(y * param['scale_y'])

        print(f"Clicked at: ({orig_x}, {orig_y})")

        # Store the coordinate
        coordinates.append((orig_x, orig_y))


# ------------------------------------------------------------
# Load the image the user will click on.
# This should be a frame from the SDC camera.
# ------------------------------------------------------------
image = cv2.imread("../frames/multi.png")
if image is None:
    print("Failed to load image. Check the file path!")
    exit()


# ------------------------------------------------------------
# Resize the image so it fits comfortably on the user's screen.
# We compute a scale factor and ensure we NEVER upscale the image.
# ------------------------------------------------------------
orig_height, orig_width = image.shape[:2]

max_width, max_height = 1400, 900   # screen-friendly size limits

# scale <= 1 ensures we only shrink, never enlarge
scale = min(max_width / orig_width, max_height / orig_height, 1)

new_width = int(orig_width * scale)
new_height = int(orig_height * scale)

resized_image = cv2.resize(image, (new_width, new_height))


# ------------------------------------------------------------
# Display the resized image and register the mouse callback.
# ------------------------------------------------------------
cv2.imshow('Image', resized_image)

# Pass scale factors so we can convert clicks back to original coordinates
cv2.setMouseCallback('Image', get_coordinates, {'scale_x': 1/scale, 'scale_y': 1/scale})

print("Click on each cone to record coordinates. Press 'q' to quit.")


# ------------------------------------------------------------
# Main loop: keep showing the image until the user presses 'q'.
# ------------------------------------------------------------
while True:
    cv2.imshow('Image', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# ------------------------------------------------------------
# Print results after the user finishes clicking.
# ------------------------------------------------------------
print("All recorded coordinates (relative to original image)")
print(coordinates)
print("Image height and width:", image.shape[:2])
