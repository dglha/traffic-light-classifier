import matplotlib.pyplot as plt  # Plotting library


def show_history(history):
    """
    Visualize the neural network model training history

    :param:history A record of training loss values and metrics values at
                   successive epochs, as well as validation loss values
                   and validation metrics values
    """
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history["loss"], label="Mất mát khi trainning")
    plt.plot(history.history["val_loss"], label="Mất mát validation")
    plt.plot(history.history["accuracy"], label="Độ chính xác khi trainning")
    plt.plot(history.history["val_accuracy"], label="Độ chính xác validation ")
    plt.title("Biểu đồ hiển thị mất mát và độ chính xác khi Training")
    plt.xlabel("Epoch #")
    plt.ylabel("Mất mát/Độ chính xác")
    plt.legend()
    plt.savefig('foo123.png')
    plt.show()


def plot_trend_by_epoch(tr_value, val_value, title, y_plot, figure):
    epoch_num = range(len(tr_value))
    plt.plot(epoch_num, tr_value, 'r')
    plt.plot(epoch_num, val_value, 'b')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_plot)
    plt.legend(['Training ' + y_plot, 'Validation ' + y_plot])
    plt.savefig(figure)
