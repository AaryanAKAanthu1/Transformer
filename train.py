from transformer import Transformer
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as functional
import math
from datasets import load_dataset

dmodel = 256
dff = 2048
nheads=8
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

dataset = load_dataset("PLMFSANA/thy-shakespeare")['train']['text']
texts = "\n".join(dataset).lower()
vocab = sorted(set(texts))
batch_size = 256
seq_len = 128
step_size = 1
char2idx = {c:i for i,c in enumerate(vocab)}
idx2char = {i:c for c,i in char2idx.items()}
data = [(texts[i:i+seq_len],texts[i+seq_len]) for i in range(0, len(texts) - seq_len, step_size)]

def train(name="transformer"):
    model = Transformer(dmodel,dff, nheads, vocab_size=len(vocab), max_seq_length=seq_len).to(device)
    warmup_steps = 4000
    criterion = nn.CrossEntropyLoss()
    prevloss = 100
    step = 0
    
    for epoch in range(epochs):
        train_loss = []
        test_loss = []
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.90, 0.98), eps=1e-9)

        model.train()
        for i in range(0, int(0.90*len(data)) - batch_size, batch_size):
            step += 1
            x_batch = torch.tensor([[char2idx.get(c, 0) for c in seq] for seq,_ in data[i:i+batch_size]], dtype=torch.long).to(device)
            y_batch = torch.tensor([[char2idx.get(c, 0) for _,c in data[i:i+batch_size]]], dtype=torch.long).squeeze(0).to(device)

            lr = (dmodel ** (-0.5)) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
            for param_groups in optimizer.param_groups:
                param_groups["lr"] = lr

            optimizer.zero_grad()
            ypred = model(x_batch)
            loss = criterion(ypred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss)

            if i%(int(len(data)/4)) == 0:
                print(f"step: {step} loss: {loss:.2f} lr:{lr}")

        model.eval()
        for i in range(int(0.90*len(data)), len(data) - batch_size, batch_size):
            with torch.no_grad():
                x_batch = torch.tensor([[char2idx.get(c, 0) for c in seq] for seq,_ in data[i:i+batch_size]]).to(device)
                y_batch = torch.tensor([[char2idx.get(c, 0) for _,c in data[i:i+batch_size]]]).squeeze(0).to(device)
                ypred = model(x_batch)
                loss = criterion(ypred, y_batch)
                test_loss.append(loss)

            if i%(int(len(data)/4)) == 0:
                print(f"step:{i+1} loss:{loss:.2f}")
        test_loss = sum(test_loss) / len(test_loss)
        if test_loss < prevloss:
            prevloss = test_loss
            torch.save(model.state_dict(), name + ".pth")
        print(f"EPOCH: {epoch+1} TRAIN_LOSS: {sum(train_loss) / len(train_loss):.2f} TEST_LOSS: {test_loss:.2f}")

    

def generate(model,text, SeqLength, name="transformer"):
    model.eval()
    context = torch.tensor([char2idx.get(c,0) for c in text])
    outtext = ""

    for c in range(SeqLength):
        logits = model(context[-seq_len:])
        probs = functional.softmax(logits, dim=-1)
        idx = probs.argmax()
        context = torch.cat((context[1:], idx)) #LEFT SHIFT and ADD NEW GENERATED CHARACTER IN CONTEXT
        outtext += idx2char.get(idx.item())

    return outtext

if __name__ == "__main__":
    epochs = 10
    choice = int(input("1. Train\n2. Generate\nEnter Choice: "))

    match choice:
        case 1:
            
                train(name="char_transformer")

        case 2:
            name = "char_transformer"
            model = Transformer(dmodel,dff, nheads, vocab_size=len(vocab), max_seq_length=seq_len).to(device)
            model.load_state_dict(torch.load(name + ".pth", weights_only=True))

            try:
                while True:
                    text = input("Enter Text: ")
                    outtext = generate(model,text)
                    print(outtext)
            except KeyboardInterrupt:
                print("USER CLOSED THE PROGRAM.")
            except Exception as e:
                print(f"ERROR: {e}")