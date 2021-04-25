#%%roc曲線繪製
#y_score = keras_model2.predict(X_test)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
#%%
df = pd.read_excel('data_testtw.xlsx')
df['text'] = df.comment.apply(lambda x: " ".join(jieba.cut(str(x))))
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df.text)
sequences = tokenizer.texts_to_sequences(df.text)
data = pad_sequences(sequences, maxlen=maxlen)
word_index = tokenizer.word_index
#embedding_dim = len(zh_model[next(iter(zh_model.vocab))])
labels = np.array(df.sentiment)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
training_samples = int(len(indices) * .8)
validation_samples = len(indices) - training_samples
X_test = data[:]
Y_test = labels[:]
#%%
model = keras.models.load_model('model.h5')
y_score=model.predict(X_test)
print(roc_auc_score(Y_test, y_score))
y_pred_keras = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)

y_pred_keras = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
#%%
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Bi-LSTM+ATT (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('roc.png')
plt.show()