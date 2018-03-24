import paddle.v2 as paddle
import os
import os.path
import random
from paddle.v2.plot import Ploter

WITH_GPU = os.getenv('WITH_GPU', '0') != '0'
paddle.init(use_gpu=WITH_GPU)

HIDDEN_NUM = [64, 32, 16]

# title_train = "Train"
title_test = "Accurate"
ploter = Ploter(title_test)

is_cla = True


class DataLoader:
    def __init__(self, data_file, float_end, process_num):
        self.file = data_file
        self.float_end = float_end
        self.lines = [l.strip().split(',') for l in open(self.file, 'r')]
        self.train_set = []
        self.test_set = []
        self.split_data(process_num)

    def split_data(self, process_num=0):
        # split_index = len(self.lines) - 15 * process_num - 30
        # self.train_set = self.lines[:split_index]
        # self.test_set = self.lines[split_index: split_index + 30]

        # self.train_set = random.sample(self.lines, int(len(self.lines)*0.67))
        # self.test_set = [line for line in self.lines if line not in self.train_set]

        split_num = int(len(self.lines) / 3)
        s0 = self.lines[0: split_num]
        s1 = self.lines[split_num: 2 * split_num]
        s2 = self.lines[2*split_num:]
        ss = [s0, s1, s2]
        self.test_set = ss[process_num]
        self.train_set = ss[(process_num+1)%3] + ss[(process_num+2)%3]

    def _train_reader(self, state):
        def reader():
            # lines = [l.strip().split(',') for l in open(self.file, 'r')]
            #index_set = self.all if state != 0 else self.train_set
            # for i in index_set:
            with open(self.file, 'r') as r:
                for line in r:
                    _data = line.strip().split(',')
                    floats = map(float, _data[:self.float_end])
                    changes = map(int, _data[self.float_end:-1])
                    if is_cla:
                        label = int(_data[-1])
                    else:
                        label = [float(_data[-1])]
                    _item = [floats] + map(list, zip(changes))
                    if state != 2:
                        _item += [label]
                    yield _item

        def reader_split():
            if state == 0:
                index_set = self.train_set
            elif state == 1:
                index_set = self.test_set
            else:
                index_set = self.lines

            for _data in index_set:
                floats = map(float, _data[:self.float_end])
                changes = map(int, _data[self.float_end:-1])
                if is_cla:
                    label = int(_data[-1])
                else:
                    label = [float(_data[-1])]
                _item = [floats] + map(list, zip(changes))
                if state != 2:
                    _item += [label]
                yield _item

        return reader_split

    def train(self):
        return self._train_reader(0)

    def test(self):
        return self._train_reader(1)

    def infer(self):
        return self._train_reader(2)


def define_input(float_num, int_num):
    floats = paddle.layer.data(
        name='floats',
        type=paddle.data_type.dense_vector(float_num)
    )

    ints = [paddle.layer.data(
        name='int%d' % i,
        type=paddle.data_type.sparse_binary_vector(2)
    ) for i in range(int_num)]

    return [floats] + ints


def line_machine(input):
    y = paddle.layer.fc(input=input, size=2, act=paddle.activation.Softmax())
    return y


def bp_network(input):
    hidden0 = paddle.layer.fc(
        input=input,
        size=HIDDEN_NUM[0],
        act=paddle.activation.Relu()
    )
    hidden1 = paddle.layer.fc(
        input=hidden0,
        size=HIDDEN_NUM[1],
        act=paddle.activation.Relu()
    )
    hidden2 = paddle.layer.fc(
        input=hidden1,
        size=HIDDEN_NUM[2],
        act=paddle.activation.Relu()
    )

    if is_cla:
        y = paddle.layer.fc(
            input=hidden2,
            size=2,
            act=paddle.activation.Softmax()
        )
    else:
        y = paddle.layer.fc(
            input=hidden2,
            size=1,
            act=paddle.activation.Linear()
        )

    return y


def save_para(para_file, parameters):
    with open(para_file, 'w') as f:
        parameters.to_tar(f)
        print "Saving parameters done..."


def train(data_file, train_no, f_num, i_num):
    model_index = os.path.basename(data_file).split('.')[0]
    para_path = './result/param/parameters_%s_%d.tar' % (model_index, train_no)
    pic_path = './result/pic/%s_%d.png' % (model_index, train_no)
    infer_path = './result/infer_%s_%d.csv' % (model_index, train_no)

    result_y = bp_network(define_input(f_num, i_num))

    if is_cla:
        label = paddle.layer.data(
            name='predict', type=paddle.data_type.integer_value(2))
        cost = paddle.layer.classification_cost(
            input=result_y,
            label=label)
    else:
        label = paddle.layer.data(
            name='predict', type=paddle.data_type.dense_vector(1))
        cost = paddle.layer.square_error_cost(
            input=result_y,
            label=label)

    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(
        learning_rate=5e-4,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        momentum=0.9)

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=optimizer
    )

    feeding = {
        'floats': 0,
    }

    i = 0
    for i in range(i_num):
        feeding['int%d' % i] = i + 1
    feeding['predict'] = i + 2

    obj = DataLoader(data_file, f_num, train_no)
    train_reader = paddle.batch(paddle.reader.shuffle(obj.train(), 1024), batch_size=500)
    test_reader = paddle.batch(obj.test(), batch_size=500)

    def event_handler_cla(event):
        global min_cost

        if isinstance(event, paddle.event.EndIteration):
            if event.pass_id % 10 == 0:
                cost_tr = 1 - event.metrics["classification_error_evaluator"]
                # ploter.append(title_train, event.pass_id, cost_tr)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)

            cost_te = 1 - result.metrics["classification_error_evaluator"]
            if cost_te > min_cost:
                min_cost = cost_te
                print "\nTest with Pass %d, cost %f, eval %f" % \
                      (event.pass_id, result.cost, cost_te)
                save_para(para_path, parameters)
                # ploter.append(title_test, event.pass_id, cost_te)

            # plot
            # if event.pass_id % 10 == 0:
            ploter.append(title_test, event.pass_id, cost_te)
            ploter.plot(pic_path)
            # obj.split_data()

    def event_handler_reg(event):
        global min_cost

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)

            cost_te = result.cost
            if cost_te < min_cost:
                min_cost = cost_te
                print "\nTest with Pass %d, cost %f, eval %f" % \
                      (event.pass_id, result.cost, cost_te)
                save_para(para_path, parameters)
                ploter.append(title_test, event.pass_id, cost_te)

            # plot
            if event.pass_id % 10 == 0:
                ploter.append(title_test, event.pass_id, cost_te)
                ploter.plot(pic_path)

    event_handler = event_handler_cla if is_cla else event_handler_reg
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=100
    )

    print "Training %s done..." % model_index
    print "Best cost %f \n" % min_cost


def infer(para_path, infer_path, obj, result_y):
    with open(para_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    with open(infer_path, 'w') as result_file:
        infer_reader = obj.infer()
        for input_x in infer_reader():
            prediction = paddle.infer(
                output_layer=result_y,
                parameters=parameters,
                input=[input_x])
            if is_cla:
                classification = int(prediction[0][1] >= 0.5)
            else:
                classification = prediction[0][0]
            result_file.write("{0}\n".format(classification))

min_cost = 0


if __name__ == "__main__":
    f_i_list = [(4, 4), (4, 4), (10, 5), (11, 4)]

    m = 2
    is_cla = True
    min_cost = 0

    train('./data/model%d.csv' % (m+1), 2,  *f_i_list[m])
