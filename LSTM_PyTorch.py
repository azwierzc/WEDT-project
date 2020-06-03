import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn import metrics, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
import nltk
import numpy as np
import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from SpamClassifierLstmLayer import SpamClassifierLstmLayer
from SpamClassifierLstmPosFull import SpamClassifierLstmPosFull
from SpamClassifierLstmPosUniversal import SpamClassifierLstmPosUniversal
from SpamClassifierSingleLstmCell import SpamClassifierSingleLstmCell
from IndexMapper import IndexMapper
from UniversalTagger import UniversalTagger
from UniversalTagger import UniversalTagger
import json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

SEQUENCE_LENGTH = 100  # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.20  # ratio of testing set
OUTPUT_SIZE = 1
# N_ITERS = 5
# EPOCHS = int(N_ITERS / (len(X_train) / BATCH_SIZE))
EPOCHS = 1
HIDDEN_DIM = 100
N_LAYERS = 2
LEARNING_RATE = 0.005

# to convert labels to integers and vice-versa
label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}


def load_data():
    texts, labels = [], []
    with open("data/SMSSpamCollection", encoding="utf8") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels


# load the data
num = 2000
X, y = load_data()
X = X[:num]
y = y[:num]

# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(X)
# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)
# convertomg to numpy arrays
X = np.array(X)
y = np.array(y)

# padding sequences at the beginning of each sequence with 0's to SEQUENCE_LENGTH
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y = [label2int[label] for label in y]
y = np.asarray(y, dtype=np.float32)

XSpamText = tokenizer.sequences_to_texts(X[y == 1])
XHamText = tokenizer.sequences_to_texts(X[y == 0])

indexes = ["Ham", "Spam"]
values = [len(XHamText), len(XSpamText)]

# Bar Chart
plt.figure()
barList = plt.bar(indexes, values, align="center", width=0.5, alpha=0.5)
plt.title('Liczba wystąpień danej klasy', fontsize=20)
plt.xlabel('Liczba wystąpień', fontsize=14)
plt.ylabel('Klasa', fontsize=14)
barList[0].set_color('darkorange')
barList[1].set_color('darkblue')
plt.show()

# split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)

split_frac = 0.5  # 50% validation, 50% test

# wordcloud for insult comments
plt.subplot(325)
subset = XSpamText
text = ["1", "2"]
wc = WordCloud(background_color="black", max_words=2000)
wc.generate(" ".join("f"))
plt.axis("off")
plt.title("Częste słowa w obraźliwych komentarzach", fontsize=20)
plt.imshow(wc.recolor(colormap='Paired_r', random_state=244), alpha=0.98)

split_frac = 0.5  # 50% validation, 50% test
split_id = int(split_frac * len(X_test))
X_val, X_test = X_test[:split_id], X_test[split_id:]
y_val, y_test = y_test[:split_id], y_test[split_id:]

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

BATCH_SIZE = int(1)  # it must be a divisor X_train and X_val
# BATCH_SIZE = int(len(X_val)/1)   # it must be a divisor X_train and X_val

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)


def get_embedding_vectors(input_tokenizer, dim=100):
    embedding_index = {}
    with open(f"data/glove.6B.{dim}d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = input_tokenizer.word_index
    new_embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            new_embedding_matrix[i] = embedding_vector

    return new_embedding_matrix


embedding_matrix = get_embedding_vectors(tokenizer)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

    data_iter = iter(train_loader)
    sample_x, sample_y = data_iter.next()

    print(sample_x.shape, sample_y.shape)

VOCAB_SIZE = len(tokenizer.word_index) + 1

model_selector = 3


def get_model(selector):
    if selector == 0:
        return SpamClassifierLstmLayer(
            vocab_size=VOCAB_SIZE,
            output_size=OUTPUT_SIZE,
            n_layers=2,
            embedding_matrix=embedding_matrix,
            embedding_size=EMBEDDING_SIZE,
            hidden_dim=HIDDEN_DIM,
            device=device,
            drop_prob=0.2
        )
    elif selector == 1:
        return SpamClassifierSingleLstmCell(
            vocab_size=VOCAB_SIZE,
            output_size=OUTPUT_SIZE,
            embedding_matrix=embedding_matrix,
            embedding_size=EMBEDDING_SIZE,
            hidden_dim=HIDDEN_DIM,
            device=device,
            drop_prob=0.2
        )
    elif selector == 2:
        return SpamClassifierLstmPosFull(
            vocab_size=VOCAB_SIZE,
            output_size=OUTPUT_SIZE,
            embedding_matrix=embedding_matrix,
            embedding_size=EMBEDDING_SIZE,
            hidden_dim=HIDDEN_DIM,
            device=device,
            index_mapper=IndexMapper(tokenizer),
            drop_prob=0.2
        )
    elif selector == 3:
        return SpamClassifierLstmPosUniversal(
            vocab_size=VOCAB_SIZE,
            output_size=OUTPUT_SIZE,
            embedding_matrix=embedding_matrix,
            embedding_size=EMBEDDING_SIZE,
            hidden_dim=HIDDEN_DIM,
            device=device,
            index_mapper=IndexMapper(tokenizer),
            drop_prob=0.2
        )
    else:
        return SpamClassifierSingleLstmCell(
            vocab_size=VOCAB_SIZE,
            output_size=OUTPUT_SIZE,
            embedding_matrix=embedding_matrix,
            embedding_size=EMBEDDING_SIZE,
            hidden_dim=HIDDEN_DIM,
            device=device,
            drop_prob=0.2
        )


model = get_model(model_selector)
model.to(device)
print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

counter = 0
print_every = len(X_val)
clip = 5
valid_loss_min = np.Inf

######################## TRAINING ###########################
# Set model to train configuration
model.train()
val_losses_vector = []
train_losses_vector = []

for i in range(EPOCHS):
    h = model.init_hidden(BATCH_SIZE)
    start_time = time.time()
    avg_loss = 0

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

        # For every (print_every) checking checking output of the model against the validation dataset
        # and saving the model if it performed better than the previous time
        if counter % print_every == 0:
            val_h = model.init_hidden(BATCH_SIZE)
            val_losses = []
            # Set model to validation configuration - Doesn't get trained here
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, EPOCHS),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

            val_losses_vector.append(np.mean(val_losses))
            train_losses_vector.append(loss.item())

            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state/state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

######################## TESTING ###########################
# Loading the best model
model.load_state_dict(torch.load('./state/state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(BATCH_SIZE)
model.eval()
test_labels_vector = []
test_pred_vector = []

for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    test_labels_vector.append(labels.item())
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    test_pred_vector.append(pred.item())
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc * 100))
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc * 100))


test_labels_vector = np.array(test_labels_vector)
test_pred_vector = np.array(test_pred_vector)

# Calculating confusion matrix
confusionMatrix = confusion_matrix(test_labels_vector, test_pred_vector)
recall = recall_score(test_labels_vector, test_pred_vector, average='macro')
precision = precision_score(test_labels_vector, test_pred_vector, average='macro')
f1 = f1_score(test_labels_vector, test_pred_vector, average='macro')
accuracy = accuracy_score(test_labels_vector, test_pred_vector)
recall1 = recall_score(test_pred_vector, test_labels_vector, average='macro')

print(confusionMatrix)
print('Average recall score: {0:0.4f}'.format(recall))
print('Average precision score: {0:0.4f}'.format(precision))
print('Average f1-recall score: {0:0.4f}'.format(f1))

cross_result = cross_validate(test_pred_vector, X_test, test_labels_vector, cv=10, return_estimator=True)
print(classification_report(cross_result, X_test, y_test))


def get_predictions(text):
    model.load_state_dict(torch.load('./state/state_dict.pt'))
    model.eval()
    h = model.init_hidden(1)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    for inputs in sequence:
        inputs = np.reshape(inputs, (1, len(inputs)))
        inputs = torch.from_numpy(inputs)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    if (pred == 0):
        return "ham"
    else:
        return "spam"


# Plot training and validation loss
val_losses_iter = np.arange(len(val_losses_vector))
train_losses_iter = np.arange(len(train_losses_vector))

plt.figure()
plt.plot(train_losses_iter, train_losses_vector, 'r', label='Training loss', )
plt.plot(val_losses_iter, val_losses_vector, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Every 1000 samples'), plt.ylabel('Values')
plt.show()

text = "Congratulations! you have won 100,000$ this week, click here to claim fast"
print(get_predictions(text))

text = "Hi man, I was wondering if we can meet tomorrow."
print(get_predictions(text))

text = "Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"
print(get_predictions(text))
