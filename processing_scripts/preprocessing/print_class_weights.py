import pickle


class_weights_dict = {}
with open("/proj/NIRAL/users/nic98/DATA/class_training_weights/pretraining_class_weights.pkl", "rb") as f:
    class_weights_dict = pickle.load(f)

for i in range(7):
    print(i, class_weights_dict[i])
