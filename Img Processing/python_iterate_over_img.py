def threshold_slow(T, image):
  # grab the image dimensions
  h = image.shape[0]
  w = image.shape[1]
  
  # loop over the image, pixel by pixel
  for y in range(0, h):
      for x in range(0, w):
          # threshold the pixel
          image[y, x] = 255 if image[y, x] >= T else 0
  # return the thresholded image
  return image
