import keras
from Utils.draw_plot import show_history

traffic_model = keras.models.load_model("traffic.h5")



show_history(traffic_model)


