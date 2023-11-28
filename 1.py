import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
#
#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2_mu = nn.Linear(256, 2)
#         self.fc2_logvar = nn.Linear(256, 2)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.relu(self.fc1(x))
#         mu = self.tanh(self.fc2_mu(x))
#         logvar = self.tanh(self.fc2_logvar(x))
#         return mu, logvar
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.fc1 = nn.Linear(2, 512)
#         self.fc2 = nn.Linear(512, 784)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         x = x.view(-1, 1, 28, 28)
#         return x
# encoder = Encoder()
# decoder = Decoder()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
#
#
# def train(encoder, decoder, num_epochs):
#     train_loss = []
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (inputs, _) in enumerate(trainloader, 0):
#             optimizer.zero_grad()
#             mu, logvar = encoder(inputs)
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             z = mu + eps * std
#             outputs = decoder(z)
#             loss_recon = criterion(outputs.view(-1,784), inputs.view(-1, 784))
#             loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#             loss = loss_recon + loss_kl
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             if i % 100 == 99:
#                 print(running_loss / 100)
#                 train_loss.append(running_loss / 100)
#                 running_loss = 0.0
#     return train_loss
# num_epochs = 10
# train_loss = train(encoder, decoder, num_epochs)
#
#
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(range(len(train_loss)), train_loss)
# plt.xlabel('Iterations')
# plt.ylabel('Reconstruction loss')
# plt.title('Reconstruction Loss')
# plt.subplot(1, 2, 2)
# plt.plot(range(len(train_loss)), train_loss)
# # plt.xlabel('Iterations')
# # plt.ylabel('KL Divergence')
# # plt.title('KL Divergence')
# # plt.tight_layout()
# # plt.show()
# #
# # z = torch.randn(10, 2)
# # generated_images = decoder(z)
# # fig, axes = plt.subplots(2, 5, figsize=(10, 4))
# # for i, ax in enumerate(axes.flat):
# #     ax.imshow(generated_images[i, 0].detach().numpy(), cmap='gray')
# #     ax.axis('off')
# # plt.show()
# import pandas as pd
# from sklearn.utils import resample
# # 读取原始数据
# df = pd.read_csv('MachineLearningCVE/all2.csv')
# # 统计每个类别的数量
# class_counts = df[' Label'].value_counts()
# # 计算每个类别应该采样的数量
# desired_size = 30000
# sampling_sizes = np.ceil((class_counts / df.shape[0]) * desired_size).astype(int)
# # 初始化采样后的数据集
# sampled_data = pd.DataFrame()
# # 对每个类别进行采样
# for class_label, sampling_size in sampling_sizes.items():
#     class_data = df[df[' Label'] == class_label]
#     sampled_class = resample(class_data, n_samples=sampling_size, replace=False, random_state=42)
#     sampled_data = pd.concat([sampled_data, sampled_class])
#
# sampled_data = sampled_data.iloc[:, 1:]
# # 保存采样后的数据集
# sampled_data.to_csv('MachineLearningCVE/resample2.csv', index=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5#[5,1,3]

hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

for i in inputs:#[1,3]
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)#i:[1,1,3]

inputs = torch.cat(inputs).view(len(inputs), 1, -1)#[5,1,3]
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

input = np.array([11, 12, 21, 22, 31, 32])
input_tensor = torch.from_numpy(input)
print(input_tensor.view(2, 3, 1))

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
## 这里只是用完全没训练过的LSTM来预测一下。

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
