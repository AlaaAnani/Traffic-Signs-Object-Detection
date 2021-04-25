from matplotlib import pyplot as plt 
import cv2
def show(rows, columns, images, titles, save=False, path=None, axis='off'):
    plt.style.use('grayscale')
    fig = plt.figure(figsize=(15, 15)) 
    k = 0
    for i in range(rows):
        for j in range(columns):
            if k == len(images):
                break
            fig.add_subplot(rows, columns, k + 1)
            plt.imshow(cv2.cvtColor(images[k],  cv2.COLOR_BGR2RGB)) 
            plt.axis(axis) 
            plt.title(titles[k])
            k += 1
    if save:
        if path is not None:
            plt.savefig(path)