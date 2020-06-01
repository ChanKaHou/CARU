import os
import nltk
import torch
import torchtext

#en->de
#python -m spacy download en_core_web_sm
#python -m spacy download de_core_news_sm
en = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='<sos>', eos_token='<eos>', lower=True, fix_length=80)
de = torchtext.data.Field(tokenize='spacy', tokenizer_language='de_core_news_sm', init_token='<sos>', eos_token='<eos>', lower=True, fix_length=80)
trainData, validData, testData = torchtext.datasets.Multi30k.splits(root='../.data/', exts=('.en', '.de'), fields=(en, de))
print('Dataset Size:', len(trainData.examples), len(validData.examples), len(testData.examples))

en.build_vocab(trainData.src, min_freq=4)
de.build_vocab(trainData.trg, min_freq=4)
print('Vocabulary Size:', len(en.vocab), len(de.vocab))
#print(de.vocab.stoi['<unk>'], en.vocab.stoi['<unk>']) #0
#print(de.vocab.stoi['<pad>'], en.vocab.stoi['<pad>']) #1
#print(de.vocab.stoi['<sos>'], en.vocab.stoi['<sos>']) #2
#print(de.vocab.stoi['<eos>'], en.vocab.stoi['<eos>']) #3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################

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

    def forward(self, word, hidden):
        feature = self.Linear(word)
        if hidden is None:
            return torch.tanh(feature)
        n = torch.tanh(self.Weight(hidden) + feature)
        l = torch.sigmoid(feature)*torch.sigmoid(self.LW(hidden) + self.LL(word))
        return torch.lerp(hidden, n, l)

########################################

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = torch.nn.Embedding(len(de.vocab), 256)
        self.RNNCell = CARUCell(256, 2048)

    def forward(self, src):
        embedded = self.Embedding(src) #[seq_length, batch_size, 256]
        hidden = None
        for word in embedded:
            hidden = self.RNNCell(word, hidden)
        return hidden #[batch_size, 2048]

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = torch.nn.Embedding(len(en.vocab), 256)
        self.RNNCell = CARUCell(256, 2048)
        self.Linear = torch.nn.Linear(2048, len(en.vocab))

    def forward(self, hidden, trg):
        feature = torch.empty([80, hidden.size(0), len(en.vocab)], device=device) #[seq_length, batch_size, len(en.vocab)]
        word = torch.full([hidden.size(0)], en.vocab.stoi['<unk>'], dtype=torch.long, device=device) #[batch_size]
        for i in range(feature.size(0)):
            embedded = self.Embedding(word) #[batch_size, 256]
            hidden = self.RNNCell(embedded, hidden)
            feature[i] = self.Linear(hidden)
            word = trg[i] if trg is not None else feature[i].argmax(-1)

        return feature

class NMT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()

    def forward(self, src, trg=None):
        hidden = self.Encoder(src)
        return self.Decoder(hidden, trg)

################################################################################

model = NMT().to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True)
if os.path.exists('Multi30k-en2de-CARUCell.pkl'):
    print("Loaded Multi30k-en2de-CARUCell.pkl")
    pkl = torch.load('Multi30k-en2de-CARUCell.pkl')
    model.load_state_dict(pkl['model.state_dict'])
    optimizer.load_state_dict(pkl['optimizer.state_dict'])
    scheduler.load_state_dict(pkl['scheduler.state_dict'])

print(model)
print(optimizer)
for parameter in model.parameters():
    print(parameter.shape)

trainItr, validItr, testItr = torchtext.data.BucketIterator.splits((trainData, validData, testData), batch_size=100, shuffle=True, device=device)
lossFunc = torch.nn.CrossEntropyLoss()

while (scheduler.last_epoch < 100):
    torch.cuda.empty_cache()

    model.train()
    with torch.enable_grad():
        trainLoss = 0.0
        for step, train in enumerate(trainItr):
            optimizer.zero_grad()

            trg = model(train.src, train.trg)
            loss = lossFunc(trg.view(-1, len(en.vocab)), train.trg.view(-1))
            print('Step: %03d' %step, 'Train Loss: %.4f' %loss.data)
            trainLoss += loss.data

            loss.backward()
            optimizer.step()
        trainLoss /= len(trainItr)
        scheduler.step(trainLoss)
        print(f'Epoch: {scheduler.last_epoch:02} | Train Loss: {trainLoss:.4f}')

    #continue
    model.eval()
    with torch.no_grad():
        validBLEU = 0.0
        for step, valid in enumerate(validItr):
            ref = [[[word.item() for word in sentence if word != en.vocab.stoi['<pad>']]] for sentence in valid.trg.transpose(0,1)]
            trg = [[word.item() for word in sentence if word != en.vocab.stoi['<pad>']] for sentence in model(valid.src).argmax(dim=-1).transpose(0,1)]
            validBLEU += nltk.translate.bleu_score.corpus_bleu(ref, trg)
        validBLEU /= len(validItr)
        print(f'Epoch: {scheduler.last_epoch:02} | Valid Bleu: {validBLEU:.4f}')

        testBLEU = 0.0
        for step, test in enumerate(testItr):
            ref = [[[word.item() for word in sentence if word != en.vocab.stoi['<pad>']]] for sentence in test.trg.transpose(0,1)]
            trg = [[word.item() for word in sentence if word != en.vocab.stoi['<pad>']] for sentence in model(test.src).argmax(dim=-1).transpose(0,1)]
            testBLEU += nltk.translate.bleu_score.corpus_bleu(ref, trg)
        testBLEU /= len(testItr)
        print(f'Epoch: {scheduler.last_epoch:02} | Test Bleu: {testBLEU:.4f}')

    #continue
    torch.save(
        {
            'model.state_dict': model.state_dict(),
            'optimizer.state_dict': optimizer.state_dict(),
            'scheduler.state_dict': scheduler.state_dict()
            },
        'Multi30k-en2de-CARUCell-%02d-(%.04f,%.04f,%.04f).pkl' %(scheduler.last_epoch, trainLoss, validBLEU, testBLEU)
        )
