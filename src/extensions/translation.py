import numpy as np
import chainer

class Translation(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_READER

    def __init__(
            self,
            model,
            translate_func,
            test_data,
            max_length=100
    ):
        self.model = model
        self.translate_func = translate_func
        self.test_data = test_data
        self.max_length = max_length
        self.data_size = len(test_data)

    def __call__(self, test_data):
        source, target = test_data[np.random.choice(self.data_size)]
        result = self.translate_func([self.model.xp.array(source)], self.max_length)[0]

        source_sentence = ' '.join(test_data.source_index2token(source)[1:-1])
        target_sentence = ' '.join(test_data.target_index2token(target)[1:-1])
        result_sentence = ' '.join(test_data.target_index2token(result)[1:-1])
        print('# source : ' + source_sentence)
        print('# target : ' + target_sentence)
        print('# result : ' + result_sentence)
