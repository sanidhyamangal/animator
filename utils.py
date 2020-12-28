"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import matplotlib.pyplot as plt  # for plotting figures


def generate_and_save_images(model, name, test_input, show_image=True):
    """
    A helper function for generating and saving images during training ops
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.0)
        plt.axis('off')

    #TODO:show image for the
    # name= 'image_at_epoch_{:04d}.png'.format(epoch)
    plt.savefig(name)
    if show_image:
        plt.show()
