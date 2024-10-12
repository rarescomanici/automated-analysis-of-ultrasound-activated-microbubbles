## HoughCircle approach, not useful yet
# final = cv2.normalize(src=mask, dst=None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # normalization
# detected_circles = cv2.HoughCircles(final,
#                                     cv2.HOUGH_GRADIENT, 1, mask.shape[0]/16, param1 = 100,
#                                     param2 = 30, minRadius = 0, maxRadius = 0)
# if detected_circles is not None:
#     # Convert the circle parameters a, b and r to integers.
#     detected_circles = np.uint16(np.around(detected_circles))
#
#     for pt in detected_circles[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
#
#         # Draw the circumference of the circle.
#         cv2.circle(processing_data[image_index], (a, b), r, (0, 255, 0), 2)
#
#         # Draw a small circle (of radius 1) to show the center.
#         cv2.circle(processing_data[image_index], (a, b), 1, (0, 0, 255), 3)
#         cv2.imshow("Detected Circle", processing_data[image_index])
#         cv2.waitKey(0)