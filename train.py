from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import os
import splitfolders
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stan.models import STAN
from stan.utils.metrics import dice_coef
from stan.utils.losses import focal_tversky_loss

from tools.helpers import get_callbacks, get_generators

def split_train_test(path):
    splitfolders.ratio(path, seed=42, ratio=(.7, .3))

def plot_dice_coef_loss_on_epochs(history):
    plt.figure(figsize=(14,10))
    plt.title("Dice_Coeff&Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.plot(history.history['loss'],label="loss")
    plt.plot(history.history['dice_coeff'],label="dice_coeff")
    plt.legend(['loss','dice_coeff'])
    plt.show()

def evaluate_model(model, test_gen):
    result = model.evaluate(test_gen)
    print(result)



def train(
    train_dir, test_dir,
    n_class, lr,
    input_shape, input_channel,
    decode_mode,
    snapshot_dir, tensorboard, tensorboard_dir,
    model_name
):
    input_shape = (input_shape, input_shape)
    epochs=1
    batch_size=8

    optimizer = Adam(learning_rate=lr)
    criterion = focal_tversky_loss(gamma=0.75)

    callbacks = get_callbacks(snapshot_dir, model_name, 
                              tensorboard, tensorboard_dir, batch_size)

    train_gen, test_gen = get_generators(train_dir, test_dir,
                                        input_shape, input_channel,
                                        horizontal_flip=True)

    model = STAN(
        n_class,
        input_shape=(input_shape[0], input_shape[1], input_channel),
        decode_mode=decode_mode,
        output_activation='sigmoid'
    )
    model.compile(optimizer=optimizer, loss=criterion, metrics=[dice_coef])
    model.fit(train_gen, batch_size=batch_size, callbacks=callbacks,
              epochs=epochs, steps_per_epoch=len(train_gen))

    evaluate_model(model,test_gen)





if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

    train_dir =  os.path.join(ROOT_DIR,"Dataset_BUSI_with_GT","train")
    test_dir = os.path.join(ROOT_DIR,"Dataset_BUSI_with_GT","test")
    n_class = 1
    epochs=50
    batch_size=8
    snapshot_dir = "."
    lr=1e-4
    input_shape=256
    input_channel= 3
    decode_mode ='transpose'
    tensorboard = True
    tensorboard_dir = "./log"
    model_name = "Test"

    train(train_dir,test_dir,n_class,lr,input_shape,input_channel,decode_mode,snapshot_dir,tensorboard,tensorboard_dir,model_name)
