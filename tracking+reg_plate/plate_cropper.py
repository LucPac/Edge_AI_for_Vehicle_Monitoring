import cv2
import numpy as np

def crop_plate(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    # Cắt biển số dựa trên tọa độ
    return image[y1:y2, x1:x2]

def align_plate(plate_img):
    # Tránh lỗi nếu ảnh cắt ra bị rỗng hoặc quá nhỏ
    if plate_img.shape[0] < 10 or plate_img.shape[1] < 10:
        return plate_img

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 1. Giảm ngưỡng Canny để nhạy hơn với ảnh nhỏ
    edges = cv2.Canny(gray, 50, 150)

    # 2. Giảm ngưỡng HoughLines xuống 30-40 (phù hợp với biển số cắt nhỏ)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=40)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi/2) * 180 / np.pi

            # 3. BỘ LỌC TỬ THẦN: Chỉ lấy những đường thẳng nằm ngang (bị nghiêng từ -30 đến 30 độ)
            # Bỏ qua hoàn toàn các đường thẳng đứng để không bị xoay lật ngang ảnh
            if -30 < angle < 30:
                angles.append(angle)

        if len(angles) > 0:
            median_angle = np.median(angles)

            # 4. Tối ưu tốc độ: Chỉ xoay nếu ảnh nghiêng hơn 2 độ
            # (nghiêng ít quá xoay sẽ làm mờ chữ do thuật toán nội suy pixel)
            if abs(median_angle) > 2:
                h, w = plate_img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1)

                # borderValue=(128, 128, 128) để các góc trống sau khi xoay có màu xám,
                # OCR sẽ không bị nhầm các góc đen thành nét chữ.
                rotated = cv2.warpAffine(plate_img, M, (w, h), borderValue=(128, 128, 128))
                return rotated

    return plate_img
