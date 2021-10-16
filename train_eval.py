# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    # model.double()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    total_batch = 0                     # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0                    # 记录上次验证集loss下降的batch数
    flag = False                        # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs), flush=True)
        scheduler.step()                # 学习率衰减

        batch_num = 0
        loss_total = 0
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            # print("[outputs] {} {}".format(outputs.type(), outputs))
            # print("[labels] {} {}".format(labels.type(), labels))
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_num += 1
            loss_total += loss.item()
            # # 每多少轮输出在训练集和验证集上的效果
            # true = labels.data.cpu()
            # predict = torch.max(outputs.data, 1)[1].cpu()
            # train_acc = metrics.accuracy_score(true, predict)

        dev_acc, dev_loss = evaluate(config, model, dev_iter)
        if dev_loss < dev_best_loss:
            print("model save ...", flush=True)
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = total_batch

            # 用 test 集 评估效果
            test(config, model, test_iter)
        else:
            improve = ''
        time_dif = get_time_dif(start_time)

        loss_avg = loss_total / float(batch_num)
        msg = 'Epoch [{0:>6}/{1:>6}] Train Loss: {2:>5.2}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5} {6}'
        print(msg.format(epoch+1, config.num_epochs, loss_avg, dev_loss, dev_acc, time_dif, improve), flush=True)
        writer.add_scalar("loss/train", loss_avg, total_batch)
        writer.add_scalar("loss/dev", dev_loss, total_batch)
        # writer.add_scalar("acc/train", train_acc, total_batch)
        writer.add_scalar("acc/dev", dev_acc, total_batch)

        model.train()

    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    # start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = '[test] Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print("===================================================")
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    print("===================================================")
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)