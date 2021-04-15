from Data import DataMatricesForTrend
from Model import CNN_20D
import os
import torch.nn as nn
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(model, DM, loss_function, epoch, optimizer):
    min_eva_loss = 10000
    for i in range(epoch):
        optimizer.zero_grad()
        data = DM.next_batch()
        x = data['data'].cuda()
        label = data['trend']
        label = torch.tensor(label, dtype=torch.long).cuda()
        total = 0
        correct = 0

        for j in range(x.shape[1]):
            x_item = x[:, j:j+1, :, :]
            label_item = label[:,j]
            output = model(x_item)
            predict = output.argmax(dim = 1)
            if j==0:
                loss = loss_function(output, label_item)
            else:
                loss = loss + loss_function(output, label_item)
            total = total + x.shape[0]
            correct = correct + (predict.cpu().numpy()==label_item.cpu().numpy()).sum()
        loss = loss / x.shape[1]
        acc = correct / total *100

        loss.backward()
        optimizer.step()

        eva_loss, eva_acc = evaluate(model, DM, loss_function)
        print("Epoch Step: %d | Train Loss: %f | Accuracy rate %f%% | Test Loss: %f | Test Accuracy rate %f%%" %
            (i, loss.item(), acc, eva_loss, eva_acc))
        if eva_loss < min_eva_loss:
            min_eva_loss = eva_loss
            torch.save(model, './checkpoint/model.pkl')
            print("save model!")

def evaluate(model, DM, loss_function):
    data = DM.get_test_set()
    x = data['data'].cuda()
    label = data['trend']
    label = torch.tensor(label, dtype=torch.long).cuda()
    loss = 0
    total = 0
    correct = 0

    for j in range(x.shape[1]):
        x_item = x[:, j:j+1, :, :]
        label_item = label[:,j]
        output = model(x_item)
        predict = output.argmax(dim = 1)
        loss_item = loss_function(output, label_item).item()
        loss = loss + loss_item
        total = total + x.shape[0]
        correct = correct + (predict.cpu().numpy()==label_item.cpu().numpy()).sum()
    loss = loss / x.shape[1]
    acc = correct / total *100
    return loss, acc



def main(epoch=12000, batch_size=32, window_size=20, trend_size=20, coin_number=10, feature_number=5,
          test_portion=0.15, portion_reversed=False, is_permed=True, buffer_bias_ratio=5e-5, lr=0.001):
    
    DM = DataMatricesForTrend(batch_size=batch_size,
                              window_size=window_size,
                              coin_number=coin_number,
                              feature_number=feature_number,
                              test_portion=test_portion,
                              trend_size=trend_size,
                              portion_reversed=portion_reversed,
                              is_permed=is_permed,
                              buffer_bias_ratio=buffer_bias_ratio)

    model = CNN_20D().cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    train(model, DM, loss_function, epoch, optimizer)

    model = torch.load('./checkpoint/model.pkl')
    loss, acc = evaluate(model, DM, loss_function)
    print("Test: | Test Loss: %f | Test Accuracy rate %f%%" %
            (loss, acc))


if __name__=='__main__':
    main()