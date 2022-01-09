from PIL import Image

size = 640, 480 # The dimensions of the webcam we're using
name = "Chair"

for i in range(200, 300):
    try:
        im = Image.open("data/" + str(i) + ".jpg")
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save("adj/" + name + str(i) + "m.jpg")
    except:
        continue