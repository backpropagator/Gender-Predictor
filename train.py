import torch
from data import *
from model import *
import random
import time
import math

n_hidden = 128
n_iter = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005



def categoryFromOutput(output):
    top_k, top_i = output.topk(1)
    category_i = top_i[0].item()
    return gender[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
    category = randomChoice(gender)
    name = randomChoice(gender_dict[category])
    category_tensor = torch.tensor([gender.index(category)], dtype=torch.long)
    name_tensor = nameToTensor(name)
    return category, name, category_tensor, name_tensor

rnn = RNN(n_letter, n_hidden, n_category)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, name_tensor):
    hidden = rnn.initHidden()
    
    rnn.zero_grad()
    
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()
    
    optimizer.step()
    
    return output, loss.item()

current_loss = 0
all_losses = []

def guessToWord(guess):
    if guess == 1:
        return "Male"
    elif guess == 0:
        return "Female"
    else:
        return "Unisexual"

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iter+1):
    category, name, category_tensor, name_tensor = randomTrainingExample()
    output, loss = train(category_tensor, name_tensor)
    
    current_loss += loss
    
    if iter%print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        gen_cat = guessToWord(guess)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iter * 100, timeSince(start), loss, name, guess, correct))
    
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


torch.save(rnn, 'gender-rnn-classification.pt')