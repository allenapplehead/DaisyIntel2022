# Do the necessary imports
import cv2
import numpy as np

""" Basement measurements
Coordinates in the real world
A (0cm, 0cm)
B (246cm, 0cm)
C (156cm, 184cm)
D (26.5cm, 184cm)
Coordinates in the camera feed
A (797, 598)
B (75, 480)
C (450, 317)
D (721, 343)
"""

# Coordinates from basement
quad_coords = {
    "lonlat": np.array([
        [0, 0], # Bottom right
        [246, 5], # Bottom left
        [154, 179], # Top left
        [25, 173] # Top right
    ]),
    "pixel": np.array([
        [797, 598], # Bottom right
        [75, 480], # Bottom left
        [450, 317], # Top left
        [721, 343] # Top right
    ])
}

# Pixel mapper from camera feed to longitutde latitude
class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T

# define a video capture object
vid = cv2.VideoCapture(0)

# CONSTANTS
VID_WIDTH = 800
VID_HEIGHT = 600

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Get vid dimensions
    ORIG_WIDTH = vid.get(3)
    ORIG_HEIGHT = vid.get(4)

    # Get scaling ratios
    SW = ORIG_WIDTH / VID_WIDTH
    SH = ORIG_HEIGHT / VID_HEIGHT

    # Draw in alignment points for 2D perspective transform from feed to floor
    cv2.circle(frame, (int(797 * SW), int(598 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(frame, (int(75 * SW), int(480 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(frame, (int(450 * SW), int(317 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.circle(frame, (int(721 * SW), int(343 * SH)), radius=2, color=(0, 0, 255), thickness=-1)
    
    pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    uv_0 = (350, 400)
    lonlat_0 = pm.pixel_to_lonlat(uv_0)
    x = uv_0[0] * SW
    y = uv_0[1] * SH
    tx = lonlat_0[0][0] * SW
    ty = lonlat_0[0][1] * SH
    # Test out the transformation
    cv2.circle(frame, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
    print("Transformed [cm]:", tx, ty)
    # Display the resulting frame
    cv2.imshow('frame', cv2.resize(frame, (VID_WIDTH, VID_HEIGHT)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
