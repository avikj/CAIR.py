from PIL import Image, ImageFilter
import numpy as np
from skimage.filters import scharr
def main():
  im = Image.open("land.jpg").convert('RGB')
  #im.show()
  imarray = np.copy(np.asarray(im))

  target_dim = imarray.shape[:2]-np.array([6, 3])
  print imarray.shape[:2], target_dim
  for i in range(imarray.shape[1]-target_dim[1]):
    print 'removing vertical seam', i
    imarray = remove_vertical_seam(imarray)
  for i in range(imarray.shape[0]-target_dim[0]):
    print 'removing horizontal seam', i
    imarray = remove_horizontal_seam(imarray)
  Image.fromarray(imarray).show()
def remove_horizontal_seam(imarray):
  print imarray.shape
  imarray = np.transpose(imarray, (1, 0, 2))
  print imarray.shape
  return np.transpose(remove_vertical_seam(imarray), (1, 0, 2))
def remove_vertical_seam(imarray):
  ''' # Compute energy 
  blurredarray = np.copy(np.asarray(Image.fromarray(imarray).filter(ImageFilter.GaussianBlur(1))))
  energy_map = np.zeros((blurredarray.shape[0], blurredarray.shape[1]))
  for i in range(blurredarray.shape[0]):
    for j in range(blurredarray.shape[1]):
      energy_map[i][j] = energy(blurredarray, i, j)
  energy_map -= np.amin(energy_map)

  print (energy_map/np.amax(energy_map))
  Image.fromarray(energy_map*255/np.amax(energy_map)).show()
  '''
  # use sobel filter for energy
  blurredarray = np.copy(np.asarray(Image.fromarray(imarray).convert('LA')))[:,:,0]
  '''dx = ndimage.scharr(blurredarray, 0).astype(np.float32)  # horizontal derivative
  dy = ndimage.scharr(blurredarray, 1).astype(np.float32)  # vertical derivative
  print dx.shape, dy.shape, np.hypot(dx, dy).shape'''
  print blurredarray.shape
  energy_map = scharr(blurredarray)# np.linalg.norm(np.hypot(dx, dy), axis=2)  # magnitude
  '''Image.fromarray((energy_map/np.max(energy_map)*255).astype('uint8')).show()
  print energy_map
  print energy_map.shape'''
  seam = min_vertical_seam(energy_map)
  '''seam_disp = np.copy(imarray)
  for i in range(len(seam)):
    imarray[i][seam[i]] = np.array([255, 0, 0])
  Image.fromarray(imarray).show()'''
  without_seam = [[] for i in range(imarray.shape[0])]
  for i in range(imarray.shape[0]):
    without_seam[i] = np.concatenate([imarray[i][:seam[i]], imarray[i][seam[i]+1:]])
  without_seam = np.array(without_seam)
  return without_seam
def energy(imarray, i, j):
  '''if i >= 1 and i < imarray.shape[0]-1:
    ider = color_diff(imarray[i-1][j], imarray[i+1][j])/2
  elif i == imarray.shape[0]-1:'''
  ider = color_diff(imarray[i-1][j], imarray[i][j])
  '''else:
    ider = color_diff(imarray[i][j], imarray[i+1][j])'''

  '''if j >= 1 and j < imarray.shape[1]-1:
    jder = color_diff(imarray[i][j-1], imarray[i][j+1])/2
  elif j == imarray.shape[1]-1:'''
  jder = color_diff(imarray[i][j-1], imarray[i][j])
  '''else:
    jder = color_diff(imarray[i][j], imarray[i][j+1])'''
  return (abs(ider)**2+abs(jder)**2)**.5
def color_diff(p1, p2):
  return abs(np.linalg.norm(p1-p2))

def min_vertical_seam(energy_map):
  # skip first and last cols
  energy_map[:,0]=energy_map[:,-1]=float('inf')
  min_energy_below = np.zeros(energy_map.shape)
  min_energy_below[energy_map.shape[0]-1] = np.copy(energy_map[energy_map.shape[0]-1])
  print energy_map.shape
  for i in range(energy_map.shape[0]-2, -1, -1):
    for j in range(energy_map.shape[1]):
      min_energy_below[i][j] = float('inf')
      if j < energy_map.shape[1]-1:
        min_energy_below[i][j] = min_energy_below[i+1][j+1]
      if j > 0:
        min_energy_below[i][j] = min(min_energy_below[i][j], min_energy_below[i+1][j-1])
      min_energy_below[i][j] = min(min_energy_below[i][j], min_energy_below[i+1][j])
      min_energy_below[i][j] += energy_map[i][j]
 # rint min_energy_below
  min_cols = []
  min_energy_seam_start = 0
  for j in range(1, energy_map.shape[1]):
    if min_energy_below[0][j] < min_energy_below[0][min_energy_seam_start]:
      min_energy_seam_start = j
  min_cols.append(min_energy_seam_start)
  for i in range(1, energy_map.shape[0]):
    min_col = min_cols[-1]
    if min_cols[-1] > 0 and min_energy_below[i][min_cols[-1]-1] < min_energy_below[i][min_col]:
      min_col = min_cols[-1]-1
    if min_cols[-1] < energy_map.shape[1]-1 and min_energy_below[i][min_cols[-1]+1] < min_energy_below[i][min_col]:
      min_col = min_cols[-1]+1
    min_cols.append(min_col)
  return min_cols
main()

