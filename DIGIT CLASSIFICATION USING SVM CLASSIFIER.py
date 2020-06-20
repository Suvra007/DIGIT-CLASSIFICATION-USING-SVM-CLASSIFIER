from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits=load_digits()
_,axes=plt.subplots(2,10)
images_and_labels=list(zip(digits.images,digits.target))
for ax,(image,label) in zip(axes[0,:],images_and_labels[:10]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    ax.set_title("Training"%label)
classifier=svm.SVC(gamma=0.001)
n_samples=len(digits.images)
data=digits.images.reshape((n_samples,-1))
X,Xt,y,yt=train_test_split(data,digits.target,test_size=0.5,shuffle=False)
classifier.fit(X,y)
pred=classifier.predict(Xt)
images_and_labels=list(zip(digits.images[n_samples//2:],pred))
for ax,(image,label) in zip(axes[1,:],images_and_labels[:10]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    ax.set_title("Prediction:%i"%label)
print("Classification report for classifier%s:\n%s\n"%(classifier,metrics.classification_report(yt,pred)))
d=metrics.plot_confusion_matrix(classifier,Xt,yt)
d.figure_.suptitle("Confusion matrix")
print("Confusion matrix:\n%s"%d.confusion_matrix)
plt.show()
