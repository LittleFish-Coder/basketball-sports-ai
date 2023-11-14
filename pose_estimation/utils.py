import numpy as np


def calculate_degree(pt1: tuple, pt2: tuple, pt3: tuple):
    """
    Calculate the angle between three points
    """
    # Convert tuple to numpy array
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt3 = np.array(pt3)

    # Calculate vectors
    vec1 = pt2 - pt1
    vec2 = pt3 - pt2

    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate angle (radian)
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))

    # Convert to degrees
    degrees = np.degrees(angle)

    # Calculate the interior angle
    interior_angle = 180 - degrees

    return interior_angle.round(2)
