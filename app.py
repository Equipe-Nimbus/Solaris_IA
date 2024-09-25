from Modelo.predict import run_predict
import pprint

#Thumbnail 
svgs = run_predict("./Modelo/checkpoints/checkpoint_epoch40.pth", "test", False, 0.5, (747, 768), False, 2)
