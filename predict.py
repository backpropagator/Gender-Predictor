from model import *
from data import *
import sys


rnn = torch.load('gender-rnn-classification.pt')

def guessToWord(guess):
    if guess == 1:
        return "Male"
    elif guess == 0:
        return "Female"
    else:
        return "Unisexual"


def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(input_line, n_predictions=2):
    print('\n> %s' % input_line)
    
    with torch.no_grad():
        output = evaluate(nameToTensor(input_line))
        
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, guessToWord(gender[category_index])))
            predictions.append([value, gender[category_index]])

if __name__ == '__main__':
    predict(sys.argv[1])