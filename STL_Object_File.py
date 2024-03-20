import numpy as np
from stl import mesh
import cv2
from typing import Tuple


class STL_Object:
    def __init__(self, file_name: str = None):
        if file_name is None:
            print("No file to load. Fix this later.")
            self.mesh: mesh = mesh.Mesh(data=[])
        else:
            try:
                self.mesh: mesh = mesh.Mesh.from_file(file_name)
            except Exception as loadingException:
                print("Can't open file: {0}.\n{1}".format(file_name, loadingException))
        num_faces: int = len(self.mesh)
        self.face_list: np.ndarray = np.ones((num_faces, 3, 4), dtype=float)
        self.face_list[:, 0, 0:3] = self.mesh[:, 0:3]
        self.face_list[:, 1, 0:3] = self.mesh[:, 3:6]
        self.face_list[:, 2, 0:3] = self.mesh[:, 6:9]

    def apply_transform(self, A: np.ndarray) -> np.ndarray:
        """
        A is a (4 x 4) transform matrix (much like the 3 x 3 matrices used for 2-d transforms)
        :param A: a numpy array with shape (4,4)
        :return: a new list of points with the same shape as self.face_list, (N,3,4)
        """
        """
        I've written this one for you because it was a bit complicated in numpy. The problem is that you have an
        array of points that you want to vector multiply by a (4 x 4) matrix, A: result = A • p for each of these 
        points, but if you try multiplying matrices with shape (4, 4) • (N, 3, 4), you get an error. So I've employed 
        a trick to turn the list into a (4, 3N) matrix, which we can multiply, and then convert this back into a 
        (N, 3, 4) matrix.
        """

        # if you have N faces, this will take the (N x 3 x 4) matrix and turn it into a (4 x 3N) matrix.
        face_t: np.ndarray = self.face_list.transpose().reshape((4, -1))
        # perform the dot product
        product: np.ndarray = A.dot(face_t)
        # turn the resulting (4 x 3N) matrix and turn it back into a resulting (N x 3 x 4) matrix.
        result: np.ndarray = product.reshape((4, 3, -1)).transpose()

        # renormalize the vector - divide the resulting (wx, wy, wz, w) matrix by w to get (x, y, z ,1)
        result[:, :, 0] = result[:, :, 0]/result[:, :, 3]
        result[:, :, 1] = result[:, :, 1]/result[:, :, 3]
        result[:, :, 2] = result[:, :, 2]/result[:, :, 3]
        result[:, :, 3] = 1

        return result

    def convert_to_2d(self, A: np.ndarray = None, d1: float = 400, d2: float = 400) -> np.ndarray:
        """
        uses a projection to find the equivalent 2-d coordinates for the given list of 3-d coordinates
        :param A: the (4 x 4) transform to apply to this 3-d shape before rendering.
        :param d1: the distance from the eye to the screen
        :param d2: the distance from the eye to the origin
        :return: an (N x 3 x 2) ndarray of 2d points corresponding to the view given.
        """
        # set A to the Identity matrix, if you haven't been given another one explicitly.
        if A is None:
            A: np.ndarray = np.identity(4, dtype=float)

        # faces is an N x 3 x 4 matrix built from the imported mesh.
        faces: np.ndarray = self.apply_transform(A)
        # ------------------------------------------------------------------------
        # TODO: you write this. Build a perspective converter (3 x 4) matrix, and multiply it by each of the transformed
        #  vectors in the list, "faces" generated just before this comment. (Base this on the apply_transform() method
        #  I wrote.) Don't forget to divide by w at the end.
        #  For each point, return just the normalized (x, y), not (x, y, 1) or (wx, wy, w).

        return np.array([[[0, 0], [1, 0], [0, 1]], ])  # replace this with your code... this is just a dummy (1 x 3 x 2)
        # ------------------------------------------------------------------------

    def find_normals(self, transform: np.ndarray = None) -> np.ndarray:
        """
        calculates the vectors that point perpendicular to each face, as determined by the cross product of the first
         two edges of each face:
         (0 --> 1) x (1 --> 2)
        :param transform: the transform to apply to the faces before calculating the norms (alternately, apply this to
        the norms after you find them.)
        :return: an N x 3 array of (x,y,z). This is not a normalized vector; it does not necessarily have a length of 1.
        """
        if transform is None:
            transform = np.eye(4, dtype=float)

        faces: np.ndarray = self.apply_transform(transform)

        # ------------------------------------------------------------------------
        # TODO: (if you are trying to do shading or back-facing culling) calculate a list of normal vectors to each
        #  face. This is done by finding the vector from point 0 - point 1 and then finding the vector from point 1
        #  to point 2. The normal is the cross product of these two vectors. You should research the cross product in
        #  numpy for this - it should not be complicated in your code!

        first_sides: np.ndarray = faces[:, 1, 0:3] - faces[:, 0, 0:3]  # I've done the first side - the result of this
        # line is an (N x 3) matrix of (dx, dy, dz).

        return first_sides  # replace this (?) with your code.
        # ------------------------------------------------------------------------

    @staticmethod
    def normalize(list_to_normalize: np.ndarray) -> np.ndarray:
        """
        multiply each of the shape(3) vectors so that it has a length of 1. This is typically done by finding the length
        of each vector and dividing the vector by that length
        NOTE: This is not to be confused with finding "normals," which is a different process!

        :param list_to_normalize: a (N x 3) list of 3-vectors
        :return: a (N x 3) list of 3-vectors each with length 1 in the same direction as the originals.
        """

        # ----------------------------------------------------------------------------
        # TODO: (if you are trying to do shading) normalize each of the N vectors. Big hint: check out the "axis"
        #  parameter in numpy.linalg.norm.

        pass
        return list_to_normalize  # currently returning UN-normalized list. Replace this with your results.
        # -----------------------------------------------------------------------------

    def find_midpoints(self, transform: np.ndarray = None) -> np.ndarray:
        """
        calculates the centers of each of the faces by averaging the three points' (x,y,z) values, to get a single
        (x, y, z) value for each face.
        :param transform: The transform to apply to the faces before averaging. (Alternately, you can find the averages
        first and then apply this transform.)
        :return: an N x 3 array of (x,y,z) points.
        """
        if transform is None:
            transform: np.ndarray = np.eye(4, dtype=float)

        faces: np.ndarray = self.apply_transform(transform)
        # ----------------------------------------------------------------------------
        # TODO: (if you are trying to do shading) find the midpoint of each face. Hint: take a look at the "axis"
        # parameter in numpy.sum.

        # replace this with your code... this is just a dummy  (1 x 3 x 3)
        return np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]], ])
        # ----------------------------------------------------------------------------

    def find_center_rays(self, transform: np.ndarray = None, d2: float = 400) -> np.ndarray:
        """
        finds the rays from the center of each face of this object to the location of the "eye" at (0,0,d2).
        :param transform: The transform to be applied to each face before you calculate the midpoints.
        :param d2: the distance from the eye to the origin, along the z-axis
        :return: an N x 3 array of (dx, dy, dz) vectors, describing the line from the center of each face to the eye.
        """
        # ----------------------------------------------------------------------------
        # TODO: (if you are trying to do shading) find the vector from the center of each face to the eye.
        # Hint: consider vector subtraction.
        pass

        # ----------------------------------------------------------------------------

    def draw_self(self,
                  window: np.ndarray,
                  transform: np.ndarray = None,
                  d1: float = 400,
                  d2: float = 400,
                  origin: Tuple[float, float] = None) -> None:
        """
        draws this object into the given (h x w x 3) array, to be displayed onscreen.
        :param window: an array with shape (h,w,3) to draw into.
        :param transform: the transform to apply to this object before drawing it.
        :param d1: distance from the "eye" to the projection screen
        :param d2: distance from the "eye" to the 3-d origin
        :param origin: the location of (0,0) on the window. By default, this will be the center of the window.
        :return: None
        """
        if origin is None:
            origin: Tuple[float, float] = (window.shape[0]/2, window.shape[1]/2)
        # ----------------------------------------------------------------------------
        # TODO: PHASE 1: (rendering) get a list of 2-d points for these faces via another method you have drawn already.
        #  Then draw the faces.

        # You might do this via three of cv.line() or one of cv.polylines()

        # TODO: PHASE 2: (back face culling) calculate the normal to each face. Use the dot product with a line in the
        #  z direction to determine whether to draw each triangle or not - NOTE: this means altering what was written
        #  in phase 1. You might wish to copy that code here and comment out the original before you make your changes.

        # TODO: PHASE 3: (shading) Use the directions of the normals, the direction from the midpoints to the eye and/or
        #  the direction of a light source (your choice). Find the relative angles between these vectors via the dot
        #  product (or just use the dot product, itself) to choose a color (or shade of grey) for this face. You will
        #  want to use cv.fillConvexPoly() for this. Filling the polys in grey and then drawing the polylines in black
        #  on top can be pretty cool.

        # TODO: PHASE 4: If you are interested, try exploring the painter's algorithm or z-indexing to handle stl files
        #  with occlusions (objects in front of one another).
