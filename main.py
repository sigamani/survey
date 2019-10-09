from model import Model

# Create instance of model class
DNN = Model()

if __name__ == '__main__':
    trained_model = DNN.train()
    tested_model = DNN.test(trained_model)
