Top_Crop = 30
Bottom_Crop = 10
# Input to model is downsampled to half width and half height
Input_Shape = (80,160,3)

from util import ModelUtil, DataUtil, VisualizeUtil
from keras.callbacks import EarlyStopping, ModelCheckpoint


ModelDir = "../data/model/"
Driving_Log_Path = "../data/driving_log.csv"
Img_Data_Dir = "../data/IMG/"
Debug_Dir = "../data/debug/"
Num_Epochs = 20
Vis = False # visualize output for debugging
Batch_size = 25
Num_Val_Samples = 10
Num_Train_Samples = 32000 # param from Mojtaba's Medium post
Model_Name = ModelDir+'july9_model.h5'
class Pipeline():
    def __init__(self):
        print("init")
        self.learning_rate = 0.001
        self.modelUtil = ModelUtil()
        self.dataUtil = DataUtil()
        self.visUtil = VisualizeUtil()

    def train(self):
        num_train_samples, num_validation_samples, train_generator, validation_generator = \
            self.dataUtil.train_val_generator(csv_path = Driving_Log_Path, image_dir=Img_Data_Dir,
                                              debug_dir = Debug_Dir, batch_size=Batch_size)

        if Vis:
            self.visUtil.vis_generator(train_generator, "remove_small_angles", save_dir=Debug_Dir)
        print("build model")
        model = self.modelUtil.create_network(Top_Crop, Bottom_Crop, Input_Shape)

        print("save to : ", Model_Name)
        callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                     ModelCheckpoint(filepath=Model_Name, monitor='val_loss', save_best_only=True)]
        print("num_train_samples",num_train_samples)
        print("num_validation_samples", num_validation_samples)

        model.fit_generator(train_generator,
                            samples_per_epoch= Num_Train_Samples,
                            validation_data=validation_generator,
                            nb_val_samples=Num_Val_Samples,
                            nb_epoch=Num_Epochs,
                            callbacks=callbacks,
                            verbose=1)
        model.save(Model_Name)


def main():
    print("main function from model.py")
    pl = Pipeline()
    pl.train()



if __name__ == '__main__':
    main()