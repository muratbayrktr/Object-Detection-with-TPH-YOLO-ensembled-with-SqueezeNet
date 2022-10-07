import torch
import os
import cv2
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torchvision import datasets
from PIL import Image
from tph_yolov5 import detect_altered
import pandas as pd
import time 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('./tph_yolov5','custom',path='eternal_eon_best_3.pt',source='local')
model.conf=0.3866
#model.iou=0.5
model.image_size=1280
model.device=0

landing_model = torch.load("squeezenet_final.pt")
landing_model = landing_model.to(device)
landing_model.eval()


test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])

def draw_rectangle(img_url,df):

    color_dict={0:(0,0,255),1:(0,255,0),2:(255,0,0),3:(255,0,255)}
    #print(img_url)
    image = cv2.imread(img_url)
    color = (255, 0, 0)
    
    for ind in df.index:
        top_left_x=int(df["xmin"][ind])
        top_left_y=int(df["ymin"][ind])
        bottom_right_x=int(df["xmax"][ind])
        bottom_right_y=int(df["ymax"][ind])
        cls=int(df["name"][ind])
        image=cv2.rectangle(image,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),color_dict[cls],thickness=3)
    #print("_inferred/"+img_url.split("/")[-1])
    cv2.imwrite("_inferred/"+img_url.split("/")[-1],image)


def draw_unavailable( img_url,top_left_x,top_left_y,bottom_right_x,bottom_right_y):
    top_left_x = int(round(top_left_x))
    top_left_y = int(round(top_left_y))
    bottom_right_x = int(round(bottom_right_x))
    bottom_right_y = int(round(bottom_right_y))
    start_point = (top_left_x,top_left_y)
    end_point = (bottom_right_x,bottom_right_y)
    image = cv2.imread(img_url)
    color = (0, 0, 255)
    cv2.line(image, start_point, end_point, color, 5)
    cv2.imwrite(img_url,image) 

# Create training and validation dataloaders
def crop(img_url,top_left_x,top_left_y,bottom_right_x,bottom_right_y):
    #print("Image to be cropped:",img_url)
    img = cv2.imread(img_url)
    #print(type(img))
    top_left_x = int(round(top_left_x))
    top_left_y = int(round(top_left_y))
    bottom_right_x = int(round(bottom_right_x))
    bottom_right_y = int(round(bottom_right_y))
    crop = img[top_left_y:bottom_right_y,top_left_x:bottom_right_x]  
    #print(img_url)
    cropped_url = '_cropped/0/cropped_'+img_url.split("/")[1]
    #print(cropped_url)
    cv2.imwrite(cropped_url,crop)
    #print("Image written to:",cropped_url)
    return cropped_url
        

def process( index):
    
    
    im_path = "frame_000" + str(127+index) +".jpg"
    # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
    Detect(im_path)
    # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.


# img_url = ./_images/frame
def Detect(img_url):
    # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
    # results = self.model.evaluate(...) # Örnektir.
    # Delete cached cropped photo
    t = time.time()
    class_dict={"0":(0,-1),"1":(1,-1),"2":(2,1),"3":(3,1)}
    #print("Detecting:",img_url)
    
    results = detect_altered.run(img_url,model,imgsz=[1280,1280],conf_thres=0.3688,device=device)
    #results = model(img_url)
    # Burada örnek olması amacıyla 20 adet tahmin yapıldığı simüle edilmiştir.
    # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.
    # Örneğin :
    # for i in results: # gibi
    #print("Result:" , results, type(results))
    #print(results.pandas().xyxy[0],type(results.pandas().xyxy[0]))
    
    df = pd.DataFrame(results,columns=["xmin","ymin","xmax","ymax","conf","name"],dtype=object)
    #print(df)
    #df.save("./_inferred")
    
    draw_rectangle(img_url,df)
    for ind in df.index:
        top_left_x=df["xmin"][ind]
        top_left_y=df["ymin"][ind]
        bottom_right_x=df["xmax"][ind]
        bottom_right_y=df["ymax"][ind]
        cls,landing_status=int(df["name"][ind]),-1

        if(cls==2 or cls==3):
            cropped_url = crop(img_url,top_left_x,top_left_y,bottom_right_x,bottom_right_y)
            image_dataset = datasets.ImageFolder('_cropped', test_transform)
            dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=True, num_workers=12)
            for img,l in dataloader:
                img = img.to(device)
                landing_out = landing_model(img)
                _, preds = torch.max(landing_out, 1)
                landing_status = (preds.detach().cpu().numpy())[0]
            
            if landing_status==0:
                #print('_inferred/'+img_url.split("/")[1],top_left_x,top_left_y,bottom_right_x,bottom_right_y)
                draw_unavailable('_inferred/'+img_url.split("/")[1],top_left_x,top_left_y,bottom_right_x,bottom_right_y)
            
            os.system("rm ./_cropped/0/*")

        d = {0:'Taşıt', 1: 'İnsan', 2: 'UAP', 3: 'UAİ'}
        print(f"DETECTION: Class:[{d[cls]}],{'Available' if 1==landing_status else ('Unavailable' if 0==landing_status else 'NO LANDING')},Coords: [{top_left_x},{top_left_y},{bottom_right_x},{bottom_right_y}]")

    t2 = time.time()
    print(f"Inference finished in [{t2-t}] seconds")

#print("-------DETECTING AVAILABLE UAP AND UAI--------")
#for i in range(4,10):
#    process(i)
#print("-------DETECTING UNAVAILABLE UAP AND UAI--------")
#for i in range(10,13):
#    process(i)


for filename in sorted(os.listdir('uai_uap_valid/')):
    f = os.path.join('uai_uap_valid', filename)
    # checking if it is a file
    if os.path.isfile(f):
        Detect(f)
        
