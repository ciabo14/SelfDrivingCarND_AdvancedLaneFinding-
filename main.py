from CarDetection.DetectionManager import DetectionManager
import cv2
import os
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
test_video_path = "./Dataset/project_video.mp4"
test_image_path = "./Dataset/test_image.jpg"

index = 0

def test_on_image():
    
    result = dm.detect_cars(cv2.cvtColor(cv2.imread(test_image_path),cv2.COLOR_BGR2RGB))
    plt.imshow(result)
    plt.show()
    
def test_on_video(dm,path,f):
    print("Running on "+f+" ...")
    clip = VideoFileClip(path+f)
    output_video = "./Dataset/videos_computed/"+f+"processed.mp4"
    output_clip = clip.fl_image(process_image)

    output_clip.write_videofile(output_video, audio=False)
    
def test_on_videos(dm):
    files = os.listdir('./Dataset/videos/')
    for f in files:
        test_on_video(dm,"./Dataset/videos/",f)


def process_image(img):

    result = dm.detect_cars(img)
    return result
    #cv2.imwrite('./imgFromStream/image{}.png'.format(index),img)
    #index += 1
    #return result

import pickle
def test_svc(dm):
    data = pickle.load(open("./classifier.p","rb"))
    svc = data["svc"]
    std = data["std"]
    
    i = 0
    import glob
    tot = len(glob.glob('./Dataset/vehiclesComplete/*.png'))
    
    for image_name in glob.glob('./Dataset/vehiclesComplete/*.png'):
        image = dm.compute_feature_vector(image_name)
        #test_features = std.transform(np.hstack((image.spatial_features, image.hist_features, image.hog_features)).reshape(1, -1))
        test_features = std.transform(image.image_features)
        #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
        test_prediction = svc.predict(test_features)
        if(test_prediction == 0):
            print(image_name,'-->',test_prediction)
        if(i%100 == 0):
            print(i,' of ', tot)
        i += 1
    
def test_on_multiple_images(dm):
    
    files = os.listdir('./Dataset/imgFromStreamSelected/3')
    for f in files:
        print(f)
        image = cv2.cvtColor(cv2.imread("./Dataset/imgFromStreamSelected/3/"+f),cv2.COLOR_BGR2RGB)

        result = dm.detect_cars(image)
        plt.imshow(result)
        plt.show()
        
if __name__ == "__main__":
    
    dm = DetectionManager()
    
    dm.train_SVC()
    #test_on_multiple_images(dm)
    #test_on_videos(dm)
    
    test_on_video(dm, "./Dataset/videos/","Untitled.mp4")
    #test_on_video(dm,"./Dataset/","project_video.mp4")
    
    #test_on_image()
    #test_svc(dm)