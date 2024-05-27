from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}
model = models.load_model("baseline_mariya.keras")
def predict_image(model,path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asarray(img)
    data = data/255

    prob = model.predict(np.asarray([data])[:1])
    max_prob = prob.max()
    y_pred = class_names[np.argmax(prob)]

    return max_prob,y_pred


image_path = "placeholder_image.png"

content = ""
prob = 0
pred = ""

index = '''
<|text-center|
<|{'logo.png'}|image|width=15vw|>

<|  |>

<|{content}|file_selector|extensions=.png|>
upload image

<|  |>

<|{image_path}|image|>

<|{pred}|>

<|' '|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

>
'''

def on_change(state,var_name,var_value):
    if var_name=="content":
        state.image_path = var_value
        max_prob,y_pred = predict_image(model,var_value)
        state.prob = int(max_prob*100)
        state.pred = "This is a ** " + y_pred + " ** " + str(state.prob) + "%"
    #print(var_name,var_value)


app = Gui(page=index)

if __name__=="__main__":
    app.run(use_reloader=True)