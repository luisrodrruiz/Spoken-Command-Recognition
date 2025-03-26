import torch
import torch.nn as nn
from dataset import AudioDataset, collate_fn
from torch.utils.data import DataLoader
from crnn import CRNNModel
import sys
import os
import argparse


class SpeechModelTrainer:
    def __init__(self, train_file, dev_file, audio_path = '', batch_size = 16, learning_rate = 0.0001, model = 'cnn', out_dir = '/tmp'):

        if model != 'crnn' and model != 'vit':
            print('ERROR. Only "cnn" or "vit" models are available')
            quit()

        train_dataset = AudioDataset(train_file, audio_path = audio_path)
            
        if model == 'crnn':    
            self.model = CRNNModel()
            input_foo_sample = torch.zeros(train_dataset.get_sample_shape()).unsqueeze(0).unsqueeze(0)
            out = self.model.cnn(input_foo_sample)
            rnn_input_size = out.shape[2]
            n_classes = train_dataset.get_num_classes()
            self.model.add_gru(rnn_input_size,n_classes)

        # else:
        #     self.model = VisionTransformer(train_dataset.get_sample_shape()[0],train_dataset.get_num_classes())
            

        self.model.cuda()
        print('model = ', self.model)
        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
        dev_dataset = AudioDataset(dev_file, audio_path = audio_path)
        self.dev_dataloader = DataLoader(dev_dataset, batch_size = 32, shuffle = False, collate_fn = collate_fn)        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = learning_rate)
        self.out_dir = out_dir

    def train_epoch(self,epoch):
        train_loss = 0.0
        dev_loss = 0.0
        train_accuracy = 0.0
        dev_accuracy = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        total_train_samples = 0 
        for batch in self.train_dataloader:
            
            features, labels,lengths = batch
            features = features.cuda()
            labels = labels.type(torch.LongTensor).cuda()
            out = self.model(features)
            loss = criterion(out,labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.detach().cpu().item()
            predictions = torch.argmax(out, dim = 1)
            train_accuracy += torch.sum(torch.eq(predictions,labels))
            total_train_samples += len(features)


        train_loss /= total_train_samples
        train_accuracy /= total_train_samples

        self.model.eval()
        total_dev_samples = 0
        for batch in self.dev_dataloader:
            features, labels, lengths = batch
            features = features.cuda()
            labels = labels.type(torch.LongTensor).cuda()
            out = self.model(features)
            loss = criterion(out,labels)
            dev_loss += loss.detach().cpu().item()
            predictions = torch.argmax(out, dim = 1)
            dev_accuracy += torch.sum(torch.eq(predictions,labels))
            total_dev_samples += len(features)

        dev_loss /= total_dev_samples
        dev_accuracy /= total_dev_samples
#        print('train_accuracy = ', train_accuracy.item())
        return train_loss, train_accuracy, dev_loss, train_accuracy    
            
    def train(self,n_epochs = 100):
        best_dev_accuracy = 0.0
        for epoch in range(n_epochs):            
            train_loss,train_accuracy,dev_loss,dev_accuracy = self.train_epoch(epoch)
            print('EPOCH %d. Train: loss = %5.5f accuracy = %5.1f.  Test:  loss = %5.5f accuracy = %5.1f' %(epoch, train_loss, 100*train_accuracy, dev_loss, 100*dev_accuracy))
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                torch.save(self.model,os.path.join(self.out_dir,'best_model.pth'))
        torch.save(self.model,os.path.join(self.out_dir,'final_model.pth'))
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file', help = 'csv file for the training dataset. Each row in the csv must have at least two columns: "filename" and "label:', required = True)
    parser.add_argument('--dev_csv_file', help = 'csv file for the dev dataset. Each row in the csv must have at least two columns: "filename" and "label:', required = True)
    parser.add_argument('--audio_path', help = 'path that will be prepend to the file names in both the train and dev csv files', required = False, default = '')
    parser.add_argument('--model', help = 'model (available models: cnn, vit) ' , default = 'crnn')
    parser.add_argument('--out_dir', help = 'directory where the model will be saved', default = '/tmp')
    args = parser.parse_args()
    trainer = SpeechModelTrainer(args.train_csv_file, args.dev_csv_file, audio_path = args.audio_path, model = args.model, out_dir = args.out_dir)
    trainer.train()
 
    
if __name__ == "__main__":
    main()
