'''
Python program source code
for research article "CARU: A Content-Adaptive Recurrent Unit for the Transition of Hidden State in NLP"

Version 1.0
(c) Copyright 2020 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The python program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The python program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
import torch
import torchtext

#python -m spacy download en_core_web_sm
Text = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True)
Label = torchtext.data.LabelField(is_target=True)

trainData, testData = torchtext.datasets.IMDB.splits(Text, Label)
print('Dataset Size:', len(trainData), len(testData)) #25000 25000

Text.build_vocab(trainData.text, min_freq=4, vectors="glove.6B.100d")
Label.build_vocab(trainData.label)
print('Text Vocabulary Size:', len(Text.vocab))
print('Label Vocabulary Size:', len(Label.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################################

class MGUCell(torch.nn.Module): #Minimal Gated Unit
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.LW = torch.nn.Linear(out_feature, out_feature)
        self.LL = torch.nn.Linear(in_feature, out_feature)
        self.Weight = torch.nn.Linear(out_feature, out_feature)
        self.Linear = torch.nn.Linear(in_feature, out_feature)

    def forward(self, word, hidden):
        if hidden is None:
            return torch.tanh(self.Linear(word))
        f = torch.sigmoid(self.LW(hidden) + self.LL(word))
        h = torch.tanh(self.Weight(f*hidden) + self.Linear(word))
        return torch.lerp(hidden, h, f)

class CARUCell(torch.nn.Module): #Content-Adaptive Recurrent Unit
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.LW = torch.nn.Linear(out_feature, out_feature)
        self.LL = torch.nn.Linear(in_feature, out_feature)
        self.Weight = torch.nn.Linear(out_feature, out_feature)
        self.Linear = torch.nn.Linear(in_feature, out_feature)

    def forward(self, word, hidden): ################################################
        feature = self.Linear(word)
        if hidden is None:
            return torch.tanh(feature)
        n = torch.tanh(self.Weight(hidden) + feature)
        l = torch.sigmoid(feature)*torch.sigmoid(self.LW(hidden) + self.LL(word))
        return torch.lerp(hidden, n, l)
    
########################################

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = torch.nn.Embedding.from_pretrained(Text.vocab.vectors) #glove.6B.100d
        self.RNNCell = CARUCell(100, 256)
        self.Linear = torch.nn.Linear(256, len(Label.vocab))

    def forward(self, text):
        embedded = self.Embedding(text) #[S, batch_size, E]
        hidden = None
        for word in embedded:
            hidden = self.RNNCell(word, hidden)
        return self.Linear(hidden)

#####################################################################################

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True)
if os.path.exists('IMDB-CARUCell.pkl'):
    print("Loaded IMDB-CARUCell.pkl")
    pkl = torch.load('IMDB-CARUCell.pkl')
    model.load_state_dict(pkl['model.state_dict'])
    optimizer.load_state_dict(pkl['optimizer.state_dict'])
    scheduler.load_state_dict(pkl['scheduler.state_dict'])

print(model)
print(optimizer)
for parameter in model.parameters():
    print(parameter.shape)

loss_func = torch.nn.CrossEntropyLoss()
trainItr, testItr = torchtext.data.BucketIterator.splits((trainData, testData), batch_size=100, shuffle=True, device=device)

while (scheduler.last_epoch < 100):
    torch.cuda.empty_cache()
 
    trainLoss = 0.0
    model.train()
    with torch.enable_grad():
        for step, train in enumerate(trainItr):
            optimizer.zero_grad()

            label = model(train.text)
            loss = loss_func(label, train.label)
            print(f'{step:05} | Train Loss: {loss.data:.4f}')
            trainLoss += loss.data

            loss.backward()
            optimizer.step()
        trainLoss /= len(trainItr)
        scheduler.step(trainLoss)
        print(f'Epoch: {scheduler.last_epoch:02} | Train Loss: {trainLoss:.4f}')

    #continue
    model.eval()
    with torch.no_grad():
        testAcc = 0.0
        for step, test in enumerate(testItr):
            label = model(test.text)
            testAcc += (label.argmax(-1) == test.label).sum()
        testAcc /= len(testData)
        print(f'Epoch: {scheduler.last_epoch:02} | Test Accuracy: {testAcc:.4f}')

    #continue
    torch.save(
       {
            'model.state_dict': model.state_dict(),
            'optimizer.state_dict': optimizer.state_dict(),
            'scheduler.state_dict': scheduler.state_dict()
           },
       'IMDB-CARUCell-%03d-(%.04f,%.04f).pkl' %(scheduler.last_epoch, trainLoss, testAcc)
       )
