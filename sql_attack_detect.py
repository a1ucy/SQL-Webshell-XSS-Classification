# web攻击分析模型（包括SQL注入、XSS、WebShell、远程命令/代码执行、CSRF等）
import warnings
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

min_freq = 3
len_text = 200
batch_size = 64
num_epochs = 10
dropout = 0.5
embedding_dim = 128
hidden_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
data_path = 'sql_dataset.csv'
df = pd.read_csv(data_path, engine='python')
df.dropna(inplace=True)

# balance data
min_len = df['label'].value_counts().min()
df_class_0 = df[df['label'] == 0].sample(min_len)
df_class_1 = df[df['label'] == 1].sample(min_len)
df = pd.concat([df_class_0, df_class_1])

def split_words(x):
    y = x.lower().split(' ')
    return y
df['feature'] = df['feature'].apply(split_words)

contents = df['feature'].values.tolist()
labels = df['label'].values.tolist()

vocab = build_vocab_from_iterator(contents, min_freq=min_freq, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def txt_to_index(x):
    result = []
    for i in x:
        idx = vocab(i)
        if len(idx) > len_text:
            idx = idx[:len_text]
        elif len(idx) < len_text:
            idx += [0] * (len_text - len(idx))
        result.append(idx)
    return result

contents = txt_to_index(contents)
contents = torch.tensor(contents)
labels = torch.tensor(labels)

X_train, X_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, shuffle=True, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True,
                            dropout=dropout, num_layers=1)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, text):
        x = self.embedding(text)
        output, (h, n) = self.lstm(x)
        # combine forward and backward features
        hidden = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        y = self.linear(hidden)

        return torch.sigmoid(y).squeeze()
        # return y.squeeze()


num_embedding = len(vocab)
model = LSTM().to(device)

# loss & optimizer
loss_fn = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

train_loss, test_loss, acc = [], [], []
# Training loop
for epoch in range(num_epochs):
    total_train_loss = []
    model.train()
    train_loop = tqdm(train_loader, desc='Train')
    for texts, labels in train_loop:
        # Forward pass
        texts = texts.to(device)
        labels = labels.to(device)
        outputs = model(texts)
        # outputs = outputs.float()
        # labels = labels.float()
        loss = loss_fn(outputs, labels)
        total_train_loss.append(loss.item())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    model.eval()
    total_val_loss = []
    all_labels = []
    all_predictions = []
    all_outputs = []
    with torch.no_grad():
        correct = 0
        total = 0
        test_loop = tqdm(test_loader, desc='Test')
        for texts, labels in test_loop:
            texts = texts.to(device)
            labels = labels.to(device)
            # print(type(texts))
            outputs = model(texts)
            # outputs = outputs.float()
            # labels = labels.float()
            loss = loss_fn(outputs, labels)
            total_val_loss.append(loss.item())
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.squeeze().cpu().numpy())

    avg_val_loss = sum(total_val_loss) / len(test_loader)
    avg_train_loss = sum(total_train_loss) / len(train_loader)
    accuracy = correct / total
    train_loss.append(avg_train_loss)
    test_loss.append(avg_val_loss)
    acc.append(1-accuracy)

    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc = roc_auc_score(all_labels, all_outputs)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Accuracy: {(accuracy):.4f}, F1 score: {f1:.4f}, AUC: {auc:.4f}')

plt.plot(train_loss)
plt.plot(test_loss)
plt.plot(acc)
plt.xlabel('Epoch')
plt.legend(['Train', 'Test', 'Acc'], loc='upper left')
plt.show()