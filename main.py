import concurrent.futures
import logging
import json
from datetime import datetime
from pathlib import Path
from time import sleep
from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    not_detected = list()
    not_sent = list()
    # Get configurations from .env file
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Declare logging configuration.
    configure_logger(team_name)
    print("Detection model loaded.")
    
    # Teams can implement their codes within ObjectDetectionModel class. (OPTIONAL)
    detection_model = ObjectDetectionModel(evaluation_server_url)

    # Connect to the evaluation server.
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)
    print("Server connection handshaked.")
    # Get all frames from current active session.
    frames_json = server.get_frames()
    
    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Run object detection model frame by frame.
    for frame in frames_json:
        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])
        # Run detection model
        predictions,status = detection_model.process(predictions,evaluation_server_url)
        # Send model predictions of this frame to the evaluation server
        if status == 200:
            result,img_not_send = server.send_prediction(predictions)
            if result.status_code != 201:
                response_json = json.loads(result.text)
                print(response_json["detail"])
                if "Your requests has been exceeded 80/m limit." in response_json["detail"]:
                    for i in [10,10,10,10,10,10]:
                        print(f"Prediction timeout,sleeping for {i} seconds...")
                        sleep(i)
                        print("Trying again...")
                        result,img_not_send = server.send_prediction(predictions)
                        if result.status_code != 201:
                            continue
                        else:
                            print("Succesful, continuing.")
                            break
                else:
                    not_sent.append(img_not_send)
                    print("Couldn't sent. Continuing.")
                    continue
        else:
            print("Couldn't utilize image. Continuing")
            not_detected.append(predictions.image_url.split("/")[-1]) # Append not detected ones
            continue

    return not_detected,not_sent
    

if __name__ == '__main__':
    not_detected,not_sent = run()
    f1 = open("not_detected.txt","w")
    f2 = open("not_sent.txt", "w") 
    f1.writelines("%s\n" % i for i in not_detected); f1.close()
    f2.writelines("%s\n" % i for i in not_sent); f2.close()