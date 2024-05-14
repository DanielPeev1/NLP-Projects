import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
            features = torch.tensor(self.extractor.get_input_representation(words, pos, state), dtype=torch.int64).cuda()
            predictions = self.model(features)
            predictions = predictions.cpu().detach().numpy()
            list_of_actions = []

            for i in range(len(predictions[0])):
                list_of_actions.append ((predictions [0][i], i))

            list_of_actions = sorted (list_of_actions)
            #print ("no")
            #print (self.output_labels)
            for i in range (len(list_of_actions) -1, -1, -1):
                #print (state.stack)
                if (list_of_actions [i][1] == 0):
                    if (len (state.buffer) > 1 or (len(state.buffer) == 1 and len (state.stack) == 0)):
                        state.shift ()
                        break
                elif (list_of_actions [i][1] < 46):
                    if (len (state.stack) > 0 and state.stack [-1] != 0):
                        state.left_arc (self.output_labels[list_of_actions [i][1] * 2 - 1][1])
                        break
                else:
                    if (len(state.stack) > 0):
                        state.right_arc(self.output_labels[(list_of_actions[i][1] - 45) * 2][1])
                        break

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
