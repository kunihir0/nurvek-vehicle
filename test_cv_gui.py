# test_cv_gui.py
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV build info (partial): \n{cv2.getBuildInformation().splitlines()[:20]}") # Print first 20 lines

# Create a black image
img = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.putText(img, 'OpenCV GUI Test', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

window_name = 'OpenCV GUI Test Window'
try:
    cv2.imshow(window_name, img)
    print(f"\nSuccessfully called cv2.imshow(). Press any key in the window to close.")
    cv2.waitKey(0) # Wait indefinitely for a key press
except cv2.error as e:
    print(f"\nError during cv2.imshow() or cv2.waitKey(): {e}")
finally:
    try:
        cv2.destroyAllWindows()
        print("cv2.destroyAllWindows() called.")
    except cv2.error as e_destroy:
        print(f"Error during cv2.destroyAllWindows(): {e_destroy}")

print("Test script finished.")