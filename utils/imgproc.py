import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools
import sklearn.metrics as skmetrics

# functions to show an image
def imsave(train_loader, Model_label):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    filename = './results/' + Model_label + '_Train_Data.jpeg'
    im.save(filename)

def resultplot(Model_label, loss_list, accuracy_list, confusdata, confustarget, namestr):
    plt.figure(Model_label, figsize=(12, 6))
    plt.subplot(121)
    plt.title('Training Loss')
    plt.plot(loss_list)
    plt.xlabel('Batch Training')
    plt.ylabel('Loss')
    plt.ylim(0, 3)
    
    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.savefig(namestr[:-2]+"png")
    #plt.show()
    plt.close()

    # confusion matrix plot and Classification Report Generation #
    # https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
    acc = skmetrics.accuracy_score(confustarget, confusdata)
    title = " Acc:"+str(np.round(acc,4))+" confusion matrix"
    classes = sorted(np.unique(confustarget))

    plt.figure(figsize=(72,32))
    plt.subplot(1,2,1)
    plot_ConfusionMatrix(skmetrics.confusion_matrix(y_true=confustarget, y_pred=confusdata), classes, title=title)
    plt.subplot(1,2,2)
    plot_ConfusionMatrix(skmetrics.confusion_matrix(y_true=confustarget, y_pred=confusdata), classes, title=title, normalize=True)
    plt.savefig(namestr[:-3] + "_ConfusionMatrix.png")
    plt.savefig(namestr[:-3] + "_ConfusionMatrix.pdf")
    #plt.show()
    plt.close()
    cl_report = open(namestr[:-3] + "_ClassificationReport.txt", "w")
    print("Classification report for " + namestr[10:-3] , skmetrics.classification_report(y_true=confustarget, y_pred=confusdata, zero_division=0), sep='\n', file=cl_report)
    cl_report.close()

# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
def plot_ConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Original source: scikit-learn documentation
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = title+" (normalized) \n"
    else:
        title = title+"\n"

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
