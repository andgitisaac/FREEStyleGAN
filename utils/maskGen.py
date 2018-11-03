import math, random
from PIL import Image, ImageDraw
import numpy as np

def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts, bound ) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory
    bound - coordinates of vertices will be clipped in this square area

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = clip(ctrX + r_i*math.cos(angle), bound[0], bound[2])
        y = clip(ctrY + r_i*math.sin(angle), bound[1], bound[3])
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points

def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x

def generateMask(size, centroid, aveRadius):
    '''Generate a mask with size H x W.

    Params:
    size - (H, W), shape of the mask array
    centroid - (x, y), coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is.

    Returns a PIL object of mask.
    '''
    leftBound, upperBound = centroid[0] - 50, centroid[1] - 50
    boundary = (leftBound, upperBound, leftBound+100, upperBound+100)

    verts = generatePolygon(ctrX=centroid[0], ctrY=centroid[1], aveRadius=aveRadius, irregularity=0.35, spikeyness=0.2, numVerts=16, bound=boundary)
    img = Image.new('L', size, 0)
    ImageDraw.Draw(img).polygon(verts, outline=None, fill=1)
    img = np.array(img, dtype='float32')
    # img.show()

    return img

def generateBatchMask(batch_size, size):
    # masks = np.zeros((batch_size, 3, size, size), dtype='uint8')
    masks = np.zeros((batch_size, size, size), dtype='float32')
    local_coordinate = []

    forbiddenWidth = 64
    for i in range(batch_size):
        aveRadius = random.randint(35, 45)
        ctrX = random.randint(0+forbiddenWidth, size-forbiddenWidth)
        ctrY = random.randint(0+forbiddenWidth, size-forbiddenWidth)

        local_coordinate.append((ctrY-forbiddenWidth, ctrX-forbiddenWidth, ctrY+forbiddenWidth, ctrX+forbiddenWidth))
        
        mask = np.expand_dims(generateMask((size, size), (ctrX, ctrY), aveRadius), axis=0)
        masks[i] = mask

        # masks[i] = np.concatenate((mask, mask, mask), axis=0) # mask[i].shape = 3 x H x W
    # print(masks.shape)
    return masks, local_coordinate


# generateMask((256, 256), (230, 230), 50)



# verts = generatePolygon(ctrX=450, ctrY=450, aveRadius=50, irregularity=0.35, spikeyness=0.2, numVerts=16)
# print(verts)

# black = (0,0,0)
# white = (255,255,255)
# im = Image.new('RGB', (500, 500), white)
# imPxAccess = im.load()
# draw = ImageDraw.Draw(im)
# tupVerts = map(tuple,verts)

# # either use .polygon(), if you want to fill the area with a solid colour
# draw.polygon( list(tupVerts), outline=black,fill=white )

# # or .line() if you want to control the line thickness, or use both methods together!
# # draw.line( tupVerts+[list(tupVerts)[0]], width=2, fill=black )

# im.show()

# now you can save the image (im), or do whatever else you want with it.

# from matplotlib import pyplot as plt
# from scipy.ndimage.filters import gaussian_filter
# mask = generateMask((128, 128), (64, 64), 25)
# blurred = gaussian_filter(mask, sigma=3)

# plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(mask)
# plt.subplot(2,1,2)
# plt.imshow(blurred)
# plt.show()