import sys, getopt
import os
from torch.utils.data import Dataset
import pandas as pd
import json
from pathlib import Path
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# 載入一個可以做中文多分類任務的模型，n_class = 3
from transformers import BertForSequenceClassification

class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]
        self.mode = mode
        load_path = './data/covid-19/'
        files = os.listdir(load_path)
        datas = []
        for name in files:
            _file = os.path.join(load_path,name)
            if os.path.exists(_file) == False:
                continue
            with open(_file,mode='r+',encoding='utf-8')as jsonfile:
                try:
                    data = json.load(jsonfile)
                    datas.append(data)
                except:
                    jsonfile.close()
                    os.remove(_file)
                    continue
        print("num of datadatas : #" + str(datas.__len__()))
        self.df = pd.DataFrame.from_dict(datas, orient='columns')
        self.len = len(self.df)
        self.label_map = {'P': 0, 'N': 1, 'C': 2}
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            text = self.df.iloc[idx, :1].values
            label_tensor = None
        else:
            text , label = self.df.iloc[idx, :].values
            if (label != 'P') and (label != 'N') and (label != 'C') :
                label = 'C'
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor(ids)
        return (tokens_tensor, label_tensor)
    
    def __len__(self):
        return self.len
# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]

    # 測試集有 labels
    if samples[0][1] is not None:
        label_ids = torch.stack([s[1] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    
    
    return tokens_tensors, label_ids

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors= data[0]
            outputs = model(input_ids=tokens_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[1]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions
def main(argv):
    opts, args = getopt.getopt(argv,"dt:",[])
    task = 0
    for opt, arg in opts:
        if opt == '-d':
            print("start DEMO .....")
            task =1
        elif opt == '-t':
            print("start testing .....")
            task = 2
        else :
            print("start training .....")
            task = 3

    if task == 2:
        PRETRAINED_MODEL_NAME = "bert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME) 
        trainset = FakeNewsDataset("train", tokenizer=tokenizer)


        # 初始化一個每次回傳 64 個訓練樣本的 DataLoader
        # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
        BATCH_SIZE = 2
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                                collate_fn=create_mini_batch)
        print("start DEMO .....")
        #test data
        NUM_LABELS = 3
        saved_model = torch.load('./mymodel/model')
        model = saved_model
        #BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

        # 讓模型跑在 GPU 上並取得訓練集的分類準確率
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        model = model.to(device)
        _, acc = get_predictions(model, trainloader, compute_acc=True)
        print("classification acc:", acc)

    
    elif task == 1:
        testdata = input('your test data : ')
        while testdata :
        #test data
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese") 
            tokens = tokenizer.tokenize(testdata) 
            input_ids = tokenizer.convert_tokens_to_ids(tokens) 
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            saved_model = torch.load('./mymodel/model')
            model = saved_model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input_ids = input_ids.to(device)
            input_ids = input_ids.unsqueeze(0)
            #model = model.to(device)
            outputs = model(input_ids=input_ids)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            if pred == 0:
                print("Positive !")
            elif  pred == 1:
                print("Negative !")
            else :
                print("Unrelated or neutral")
            testdata = input('your test data : ')
    else :
        PRETRAINED_MODEL_NAME = "bert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME) 
        trainset = FakeNewsDataset("train", tokenizer=tokenizer)


        # 初始化一個每次回傳 64 個訓練樣本的 DataLoader
        # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
        BATCH_SIZE = 2
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                                collate_fn=create_mini_batch)
        # training mode
        # 訓練模式
        model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
        model.train()

        # 使用 Adam Optim 更新整個分類模型的參數
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


        EPOCHS = 6  # 幸運數字
        for epoch in range(EPOCHS):
            
            running_loss = 0.0
            for data in trainloader:
                
                tokens_tensors,labels = [t.to(device) for t in data]

                # 將參數梯度歸零
                optimizer.zero_grad()
                
                # forward pass
                outputs = model(input_ids=tokens_tensors,  labels=labels)

                loss = outputs[0]
                # backward
                loss.backward()
                optimizer.step()


                # 紀錄當前 batch loss
                running_loss += loss.item()
                
            # 計算分類準確率
            _, acc = get_predictions(model, trainloader, compute_acc=True)

            print('[epoch %d] loss: %.3f, acc: %.3f' %
                (epoch + 1, running_loss, acc))

        torch.save(model, './mymodel/model')
    #saved_model = torch.load('path/to/model')
if __name__ == "__main__":
   main(sys.argv[1:])