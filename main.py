from model import Model

# Create instance of model class
model = Model()

if __name__ == '__main__':
    trained_model = model.train()
    tested_model = model.test(trained_model)