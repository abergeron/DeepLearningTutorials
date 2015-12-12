from threading import Thread
import numpy
import time

import lstm
import channel


class LSTMLieutenant(channel.Lieutenant):
    def __init__(self, max_mb, ydim, patience):
        channel.Lieutenant.__init__(self, port=5566, cport=5567)
        self.max_mb = max_mb
        self.ydim = int(ydim)
        self.patience = patience

        self.uidx = 0
        self.eidx = 0
        self.history_errs = []
        self.bad_counter = 0

        self.stop = False

    def handle_control(self, req):
        if req == 'next':
            if self.stop:
                return 'stop'
            return 'train'
        if req == 'ydim':
            return self.ydim
        if isinstance(req, dict):
            if 'done' in req:
                self.uidx += req['done']
                if self.uidx > self.max_mb:
                    self.stop = True
            if 'valid_err' in req:
                valid_err = req['valid_err']
                test_err = req['test_err']
                self.history_errs.append([valid_err, test_err])
                harr = numpy.array(self.history_errs)[:, 0]
                if valid_err <= harr.min():
                    self.bad_counter = 0
                    return 'best'
                if (len(self.history_errs) > self.patience and
                        valid_err >= harr[:-self.patience].min()):
                    self.bad_counter += 1
                    if self.bad_counter > self.patience:
                        self.stop = True
                        return 'stop'

def lstm_control(dataset='imdb',
                 patience=10,
                 test_size=-1,
                 n_words=10000,
                 maxlen=100,
                 dispFreq=10,
                 max_epochs=5000,
                 validFreq=370,
                 saveFreq=1110,
                 batch_size=16,
                 valid_batch_size=64,
                 saveto=None,
                 ):

    l = LSTMLieutenant(max_mb=0, ydim=0, patience=patience)

    load_data, prepare_data = lstm.get_dataset(dataset)

    print "Loading data"

    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                   maxlen=maxlen)

    del valid
    del test

    l.ydim = int(numpy.max(train[1]) + 1)

    print "%d train examples" % len(train[0])

    l.max_mb = ((len(train[0]) * max_epochs) // batch_size) + 1

    def send_mb():
        while True:
            kf = lstm.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            for _, train_index in kf:
                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                x, mask, y = prepare_data(x, y)

                l.send_mb([x, mask, y])

    t = Thread(target=send_mb)
    t.daemon = True
    t.start() 
    print "Lieutenant is ready"
    start_time = time.time()
    l.serve()
    stop_time = time.time()
    print "Training time %fs" % (stop_time - start_time,)

if __name__ == '__main__':
    lstm_control()
