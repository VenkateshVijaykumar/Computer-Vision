import cv2
import numpy as np

def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    height, width = image.shape[:2]
    pts = [(0,0),(0,int(height-1)),(int(width-1),0),(int(width-1),int(height-1))]
    return pts

def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    centers = dst_points
    pts = src_points
    p = []
    for i in range(len(src_points)):
        P29a = [-pts[i][0],-pts[i][1],-1,0,0,0,pts[i][0]*centers[i][0],pts[i][1]*centers[i][0],centers[i][0]]
        P29b = [0,0,0,-pts[i][0],-pts[i][1],-1,pts[i][0]*centers[i][1],pts[i][1]*centers[i][1],centers[i][1]]
        p.append(P29a)
        p.append(P29b)
    p.append([0,0,0,0,0,0,0,0,1])
    pmx = np.matrix((p))
    b = np.matrix(([0],[0],[0],[0],[0],[0],[0],[0],[1]))
    hmx = np.linalg.solve(pmx,b)
    hmx_list  = hmx.tolist()
    hmx_list = [e for r in hmx_list for e in r]
    h_matrix = np.matrix(([hmx_list[0],hmx_list[1],hmx_list[2]],[hmx_list[3],hmx_list[4],hmx_list[5]],[hmx_list[6],hmx_list[7],hmx_list[8]]))
    h_matrix = np.asarray(h_matrix)
    return h_matrix

def project_imageA_onto_imageB(projection_corners, imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    #centers = find_markers(imageB)
    centers = projection_corners
    h_inv = np.linalg.inv(homography)
    adheight, adwidth = imageA.shape[:2]
    poly_pts = []
    poly_pts.append(list(centers[0]))
    poly_pts.append(list(centers[1]))
    poly_pts.append(list(centers[3]))
    poly_pts.append(list(centers[2]))
    poly_pts = np.array(poly_pts)
    cv2.fillPoly(imageB, np.int32([poly_pts]), (0,255,0))
    green = np.array([0,255,0], dtype = np.uint8)
    for j in range(imageB.shape[0]):
        for i in range(imageB.shape[1]):
            mat = np.matrix(([i],[j],[1]))
            new_pt = np.floor(h_inv.dot(mat))
            new_pt = new_pt.astype(int)
            if new_pt[0] > (adwidth-1):
                new_pt[0] = (adwidth-1)
            if new_pt[1] > (adheight-1):
                new_pt[1] = (adheight-1)
            if new_pt[0] < 0:
                new_pt[0] = 0
            if new_pt[1] < 0:
                new_pt[1] = 0
            new_val = imageA[new_pt[1],new_pt[0]].tolist()
            new_val = [e for r in new_val for e in r]
            new_val = [e for r in new_val for e in r]
            #print "new_val", new_val
            if np.array_equal(imageB[j,i], green):
                imageB[j,i] = new_val #adimage[new_pt[1],new_pt[0]]
            else:
                pass
    return imageB

projection = cv2.imread('project.jpg')
scene = cv2.imread('Canvas.png')
markers = [(197,124),(184,710),(1168,241),(1168,590)]
src_points = get_corners_list(projection)
homography = find_four_point_transform(markers, src_points, markers)
projected_img = project_imageA_onto_imageB(projection, scene, homography)
cv2.imwrite('projected_img.png', projected_img)