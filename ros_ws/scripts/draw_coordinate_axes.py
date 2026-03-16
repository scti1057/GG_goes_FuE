@staticmethod
def _draw_coordinate_axes(cv_img):
    h, w = cv_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compact panel in the upper-right corner.
    margin = 10
    panel_w, panel_h = 80, 85
    panel_tl = (max(0, w - panel_w - margin), margin)
    panel_br = (min(w - 1, panel_tl[0] + panel_w), min(h - 1, panel_tl[1] + panel_h))

    # Origin is near the lower-right inside this panel so +x left and +y up are visible.
    origin = (panel_br[0]-50, panel_br[1] - 18)
    axis_len = 30

    # Background panel for readability.
    overlay = cv_img.copy()
    cv2.rectangle(overlay, panel_tl, panel_br, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, cv_img, 0.65, 0, cv_img)

    color_x = (0, 0, 255)
    color_y = (0, 255, 0)
    color_z = (255, 0, 0)

    # +x points to the left in the image.
    cv2.arrowedLine(
        cv_img,
        origin,
        (origin[0] + axis_len, origin[1]),
        color_x,
        1,
        cv2.LINE_AA,
        tipLength=0.3,
    )
    cv2.putText(
        cv_img,
        '+x',
        (origin[0] + axis_len, origin[1] - 2),
        font,
        0.38,
        color_x,
        1,
        cv2.LINE_AA,
    )

    # +y points upwards in the image.
    cv2.arrowedLine(
        cv_img,
        origin,
        (origin[0], origin[1] - axis_len),
        color_y,
        1,
        cv2.LINE_AA,
        tipLength=0.3,
    )
    cv2.putText(
        cv_img,
        '+y',
        (origin[0] + 4, origin[1] - axis_len - 4),
        font,
        0.38,
        color_y,
        1,
        cv2.LINE_AA,
    )

    # +z points into the image plane (circle with cross) at the axis origin.
    z_center = origin
    z_r = 5
    cv2.circle(cv_img, z_center, z_r, color_z, 1, cv2.LINE_AA)
    cv2.line(
        cv_img,
        (z_center[0] - 3, z_center[1] - 3),
        (z_center[0] + 3, z_center[1] + 3),
        color_z,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        cv_img,
        (z_center[0] - 3, z_center[1] + 3),
        (z_center[0] + 3, z_center[1] - 3),
        color_z,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        cv_img,
        '+z',
        (z_center[0] - 20, z_center[1] + 12),
        font,
        0.38,
        color_z,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        cv_img,
        'Koordsystem',
        (panel_tl[0] + 6, panel_tl[1] + 14),
        font,
        0.35,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )