from asyncio.log import logger
import logging
import time
import os
import json
from matplotlib import image
import requests
import torch
import PIL
import cv2
from torchvision import transforms, datasets
from tph_yolov5 import detect_altered
import pandas as pd



from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject

class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        # Delete cached cropped photo in any case
        self.clear_cache()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.
        self.model = torch.hub.load('./tph_yolov5','custom',path='eternal_eon_best_3.pt',source='local')
        self.model.conf=0.3866
        self.model.image_size=1280
        self.model.device=0
        self.model.eval()

        self.landing_model = torch.load("squeezenet_final.pt")
        self.landing_model = self.landing_model.to(self.device)
        self.landing_model.eval()

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])

    def clear_cache(self):
        os.system('rm -r ./_cropped/0/*')


    def draw_rectangle(self,img_url,df):

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


    def draw_unavailable(self, img_url,top_left_x,top_left_y,bottom_right_x,bottom_right_y):
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

    def crop(self, img_url,top_left_x,top_left_y,bottom_right_x,bottom_right_y):
        #print("Image to be cropped:",img_url)
        img = cv2.imread(img_url)
        #print(type(img))
        top_left_x = int(round(top_left_x))
        top_left_y = int(round(top_left_y))
        bottom_right_x = int(round(bottom_right_x))
        bottom_right_y = int(round(bottom_right_y))
        crop = img[top_left_y:bottom_right_y,top_left_x:bottom_right_x]  
        cropped_url = './_cropped/0/cropped_'+img_url.split("/")[2]
        #print(cropped_url)
        #print("Image written to:",cropped_url)
        cv2.imwrite(cropped_url,crop)
        return cropped_url

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')

        return image_name

    def process(self, prediction,evaluation_server_url):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        image_name = self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images/")
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        im_path = "./_images/" + image_name
        
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        t1 = time.perf_counter()
        frame_results,status = self.detect(prediction, im_path)
        t2 = time.perf_counter()
        logging.info(f'{im_path} - Inference Finished in ({t2 - t1}) seconds')
        #print(f'{im_path} - Inference Finished in ({t2 - t1}) seconds')
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        #return frame_results
        return frame_results, status


    def detect(self, prediction, img_url):
        # Run inference

        class_dict={"0":(0,-1),"1":(1,-1),"2":(2,1),"3":(3,1)}
        #print("Detecting:",prediction.image_url)
        status = 200
        try:
            results = self.model(img_url)
        except:
            status = 400
            return prediction,status
        
        results = detect_altered.run(img_url,self.model,imgsz=[1280,1280],conf_thres=0.3688,device=self.device)
        df = pd.DataFrame(results,columns=["xmin","ymin","xmax","ymax","conf","name"],dtype=object)

        #df=results.pandas().xyxy[0]
        #results.save("./_inferred")
        
        self.draw_rectangle(img_url,df)

        for ind in df.index:
            top_left_x=df["xmin"][ind]
            top_left_y=df["ymin"][ind]
            bottom_right_x=df["xmax"][ind]
            bottom_right_y=df["ymax"][ind]
            cls,landing_status=int(df["name"][ind]),-1

            if(cls==2 or cls==3):
                cropped_url = self.crop(img_url,top_left_x,top_left_y,bottom_right_x,bottom_right_y)
                image_dataset = datasets.ImageFolder('_cropped', self.test_transform)
                dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)
                for img,l in dataloader:
                    img = img.to(self.device)
                    landing_out = self.landing_model(img)
                    _, preds = torch.max(landing_out, 1)
                    landing_status = (preds.detach().cpu().numpy())[0]
                self.clear_cache()

            #if landing_status==0:
                #print('_inferred/'+img_url.split("/")[1],top_left_x,top_left_y,bottom_right_x,bottom_right_y)
                #self.draw_unavailable('_inferred/'+img_url.split("/")[1],top_left_x,top_left_y,bottom_right_x,bottom_right_y)
            
            
            d_obj = DetectedObject(cls,
                                    landing_status,
                                    top_left_x,
                                    top_left_y,
                                    bottom_right_x,
                                    bottom_right_y)
            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)
        

        return prediction,status
