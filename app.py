from keras.models import load_model
import gradio as gr
import numpy as np

def classify_image(img):
    img_3d=img.reshape(1,28,28,1)
    im_resize=img_3d/255.0
    prediction=digit_model.predict(im_resize)
    pred=np.argmax(prediction)
    return pred

digit_model=load_model('D:\GitRepo\CNNDigit-Classifier-GRADIO\digit_recognition_model.h5')

label=gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(classify_image,inputs="sketchpad", outputs="label")
interface.launch(share=True)



