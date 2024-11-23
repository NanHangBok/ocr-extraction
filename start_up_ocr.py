from PIL import Image
import pytesseract
import cv2
import numpy as np
import re

def add_text_to_image(image, text, position=(50, 150), font_scale=1, color=(0, 0, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    marked_image = image.copy()
    cv2.putText(marked_image, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return marked_image

# 인식한 글자 치환 전 전처리 ( 불용어 처리 )
def clean_str(text):
    if not isinstance(text, str):
        return ''
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern, '', text)
    text = re.sub('[-=+,#/\?:^$@©*\"※—&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》«_a-zA-Z]','', text)  # 특수기호제거
    return text

# 텍스트 정제 함수
def clean_text(text):
    # 라인 단위로 처리
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 띄어쓰기 제거
        line = line.replace(' ', '')
        # 2개 이상의 점을 제거하고, 점 뒤에 숫자가 있는 경우만 유지
        line = re.sub(r'\.{2,}', '.', line)
        line = re.sub(r'\.(?!\d)', '', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# 텍스트를 라인 단위로 수정하는 함수
def process_extracted_text(text):
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        # 띄어쓰기 제거
        line = line.replace(' ', '')
        # .이 여러 개 있는 경우 처리
        parts = line.split('.')
        if len(parts) > 2:
            if parts[-1].isdigit():
                line = ''.join(parts[:-1]) + '.' + parts[-1]
            else:
                line = '.'.join(parts[:-1]) + parts[-1]
        processed_lines.append(line)
    return '\n'.join(processed_lines)

# 이미지 전처리 함수
def image_preprocessing(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    median_blurred = cv2.medianBlur(gray, 7)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(median_blurred, cv2.MORPH_OPEN, kernel)
    _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150)
    morph_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return morph_close

# 컨투어 찾기 함수
def find_contours(image, min_area_threshold):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    return filtered_contours

# 4점 변환 함수
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# 텍스트 추출을 위한 전처리 함수
def text_extraction_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to remove noise and enhance the text
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary

# 숫자만 추출하는 함수 (소수점 포함)
def extract_numbers(text):
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    return [num for num in numbers if re.match(r'^\d+(\.\d+)?$', num)]

# 실수형 숫자만 추출하는 함수
def extract_floats(numbers):
    return [num for num in numbers if '.' in num]

# 메인 함수
def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return
    
    processed_image = image_preprocessing(image)
    
    contours = find_contours(processed_image, 30)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    found_contour = False

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            warped = four_point_transform(image, approx.reshape(4, 2))
            found_contour = True
            break
    
    if not found_contour:
        print("No valid contour with four points found.")
        return  # 유효한 컨투어 없으면 종료
    
    # 이미지 크기 변경 (가로 1080, 세로 비율 유지)
    height, width = warped.shape[:2]
    new_width = 1080
    new_height = int((new_width / width) * height)
    resized_warped = cv2.resize(warped, (new_width, new_height))
    
    x, y, w, h = roi
    resized_warped = resized_warped[y:y+h, x:x+w]
    
    if resized_warped.size == 0:
        print("Cropped image is empty. Please check the ROI values.")
        return

    # 텍스트 추출을 위한 전처리
    processed_resized_warped = text_extraction_preprocessing(resized_warped)
    
    if processed_resized_warped is not None:
        contours = find_contours(processed_resized_warped, 30)
        
        # 컨투어를 사각형으로 그리기
        contour_image = resized_warped.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Tesseract 설정을 최적화
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(processed_resized_warped, lang='kor+eng', config=custom_config)
        print("추출 텍스트: ", extracted_text)
        
        cleaned_text = clean_text(extracted_text)
        processed_text = process_extracted_text(cleaned_text)
        print("정제된 텍스트: ", processed_text)
        
        numbers = extract_numbers(processed_text)
        print("숫자 추출: ", numbers)
        
        float_numbers = extract_floats(numbers)
        print("실수 추출:", float_numbers)
        
    print("*-" * 50)
    print("최종 추출물:")
    for number in float_numbers:
        print(number)
    
    # 추출한 실수들을 이미지 위에 삽입
    if float_numbers:
        combined_text = " | ".join(float_numbers)  # 실수들을 연결된 문자열로 생성
        marked_image = add_text_to_image(resized_warped, combined_text, position=(300, 100))
    else:
        marked_image = add_text_to_image(resized_warped, "No numbers extracted", position=(50, 100))
    
    # 이미지 출력
    cv2.imshow("Image with Extracted Numbers", marked_image)
    
    # 이미지 저장
    output_path = "./inbody/inbody_result_4.jpg"  # 저장할 파일 경로 및 이름
    cv2.imwrite(output_path, marked_image)
    print(f"Image saved to {output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Tesseract 경로 설정
    image_path = 'C:/Users/cczzs/Downloads/inbody/inbody_4.jpg'
    
    roi = (40, 442, 596, 162)
    
    main(image_path)
