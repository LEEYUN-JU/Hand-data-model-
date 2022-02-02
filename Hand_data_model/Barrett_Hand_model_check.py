from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

folder_data = ["All"]
# folder_data = ["Finger", "Hand", "Angle", "Effort", "All"]
normalization = ["Normalization"]
# normalization = ["Standard", "Minmax", "Robust", "Normalization"]

defalut = "D:/Classification"

Accuracy = []
temp = ""

for f, number in enumerate(range(len(folder_data))):
    
    # if f == 0: image_w = 8; image_h = 9;
                    
    # if f == 0: image_w = 8; image_h = 12;         
        
    # if f == 0: image_w = 9; image_h = 11; 
        
    # if f == 0: image_w = 9; image_h = 11;    
        
    if f == 0: image_w = 6; image_h = 17; 
    
    for i, title in enumerate(range(len(normalization))):
        caltech_dir = defalut + "/Barrett_Hand_pngs/%s/%s/test/"%(folder_data[number], normalization[title])    
    
        pixels = image_h * image_w * 3
    
        folder = os.listdir(caltech_dir)
        folder_list = [file for file in folder]
    
        Predic = np.zeros((8), dtype=int)
        Acc = np.zeros((8), dtype=float)
    
        test = 0;
    
        for i in range(len(folder_list)):
            check_model_dir = caltech_dir+folder_list[i]
        
            X = []
    
            filenames = []
    
            index = i
        
            file_name = ""
    
            files = glob.glob(check_model_dir+"/*.png*")
            for i, f in enumerate(files):
                img = Image.open(f)
                img = img.convert("RGB")
                img = img.resize((image_w, image_h))
                data = np.asarray(img)
                filenames.append(f)
                X.append(data)        
       
            X = np.array(X)
        
            model = load_model(defalut + '/Barrett_Hand_codes/models/model_%s_%s.model'%(folder_data[number], normalization[title]))
    
            prediction = model.predict(X)
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            cnt = 0
            #"ball", "banana", "can","cube","pear","spam","strawberry","tennis"
            count = np.zeros((8, 8), dtype=int)# 빈틀 만들기    
        
            for p in range(0,8):        
                for j in prediction:
                    pre_ans = j.argmax()  # 예측 레이블
                    #print(i)
                    #print(pre_ans)
                    pre_ans_str = ''
                    if pre_ans == 0: count[p][0] += 1
                    elif pre_ans == 1: count[p][1] += 1
                    elif pre_ans == 2: count[p][2] += 1
                    elif pre_ans == 3: count[p][3] += 1
                    elif pre_ans == 4: count[p][4] += 1
                    elif pre_ans == 5: count[p][5] += 1
                    elif pre_ans == 6: count[p][6] += 1
                    else: count[p][7] += 1
        
            print(count[p])
            Accuracy.append(count[p])
            Predic[test] = count[test][test]    
            test+=1
     
        for h in range(0,8):
            Acc[h] = (Predic[h] / 150)*100
            
        print((folder_data[number]), (normalization[title]), (round(np.mean(Acc),2)))
        print("")
        
        temp = folder_data[number] + "_" + normalization[title] + ": " +"%s"%(round(np.mean(Acc),2)) + "%"
        Accuracy.append(temp)        
        Accuracy.append("")
        
np.savetxt('%s.txt'%folder_data[number], Accuracy, fmt='%s', delimiter = ' ')