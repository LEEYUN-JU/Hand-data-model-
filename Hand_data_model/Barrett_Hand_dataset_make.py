from PIL import Image
import glob, numpy as np
from sklearn.model_selection import train_test_split

# folder = ["Finger", "Hand", "Angle", "Effort", "All"]
folder = ["All"]
normalization = ["Standard", "Minmax", "Robust", "Normalization"]

for f, number in enumerate(range(len(folder))):
    
    # if f == 0: image_w = 8; image_h = 9;
                    
    # if f == 1: image_w = 8; image_h = 12;         
        
    # if f == 2: image_w = 9; image_h = 11; 
        
    # if f == 3: image_w = 9; image_h = 11;    
        
    # if f == 4: image_w = 6; image_h = 17;

    if f == 0: image_w = 600; image_h = 1700;         
    
    for i, title in enumerate(range(len(normalization))):
        caltech_dir = "C:/Users/Robot 7113/Desktop"
        #caltech_dir = "D:/Classification/Barrett_Hand_pngs/%s/%s/train"%(folder[number], normalization[title])
        categories = ["ball", "banana", "can","cube","pear","spam","strawberry","tennis"] #하위 폴더명들
        nb_classes = len(categories) #카테고리 개수
    
        pixels = image_h * image_w * 3 #픽셀 수 
    
        X = []
        y = []
    
        for idx, cat in enumerate(categories):
        
            #one-hot 돌리기.
            label = [0 for i in range(nb_classes)]
            label[idx] = 1
    
            image_dir = caltech_dir + "/" + cat #폴더 찾기
            files = glob.glob(image_dir+"/*.png") #이미지 불러오기
            print(cat, " 파일 길이 : ", len(files)) #파일 개수 세기
            for i, f in enumerate(files):
                img = Image.open(f)
                img = img.convert("RGB")
                img = img.resize((image_w, image_h)) #이미지를 10*10사이즈로 전환
                data = np.asarray(img) #np.asarray(원본) np.array(복사본 생성)
    
                X.append(data) 
                y.append(label) 
            
                if i % 700 == 0: #파일 길이, 경로, 개수 출력
                    print(cat, " : ", f)
    
        X = np.array(X)
        y = np.array(y)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #test_size default = 0.25
        xy = (X_train, X_test, y_train, y_test) #데이터와 레이블의 순서쌍 유지
        #np.save("D:/Classification/Barrett_Hand_codes/dataset/dataset_%s_%s.npy"%(folder[number], normalization[title]), xy) 
        
        print("ok", len(y))
        print(X_train.shape)