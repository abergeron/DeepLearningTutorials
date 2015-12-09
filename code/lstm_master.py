import numpy

import lstm
import channel


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

    l = channel.Lieutenant(port=5566, cport=5567)

    load_data, prepare_data = lstm.get_dataset(dataset)

    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                   maxlen=maxlen)

    del valid
    del test

    if test_size > 0:
            # The test set is sorted by size, but we want to keep random
            # size example.  So we must select a random selection of the
            # examples.
            idx = numpy.arange(len(test[0]))
            numpy.random.shuffle(idx)
            idx = idx[:test_size]
            test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1

    l.register_control('ydim', lambda: str(ydim))

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

#    kf_valid = lstm.get_minibatches_idx(len(valid[0]), valid_batch_size)
#    kf_test = lstm.get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
#    print "%d valid examples" % len(valid[0])
#    print "%d test examples" % len(test[0])

    stop_flag = [False]
    save_flag = [False]
    valid_flag = [False]
    bad_counter = [0]

    def control(req):
        if req == 'next':
            if stop_flag[0]:
                return 'stop'
            if valid_flag[0]:
                valid_flag[0] = False
                return 'valid'
            return 'train'
        if isinstance(req, dict):
            if 'valid_err' in req:
                valid_err = req['valid_err']
                test_err = req['test_err']
                history_errs.append([valid_err, test_err])
                if valid_err <= numpy.array(history_errs)[:, 0].min():
                    bad_counter[0] = 0
                    return 'best'
                if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                    bad_counter[0] += 1
                    if bad_counter[0] > patience:
                        stop_flag[0] = True
            return None

    l.handle_control = control

    history_errs = []

    uidx = 0

    for eidx in xrange(max_epochs):
        kf = lstm.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
        for _, train_index in kf:
            # Select the random examples for this minibatch
            y = [train[1][t] for t in train_index]
            x = [train[0][t] for t in train_index]

            x, mask, y = prepare_data(x, y)

            l.send_mb([x, mask, y])
            uidx += 1
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                save_flag[0] = True
            if numpy.mod(uidx, validFreq) == 0:
                valid_flag[0] = True
