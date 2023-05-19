import torch
import torch.utils.data as Data
from readdata import data_reader


def get_data_loader(config, num_of_questions, max_step, fold):
    handle = data_reader(config ,fold,'../data/DKTFeatures_'+str(config.assignment)+'/train_firstatt_'+str(fold)+'.csv',
                        '../data/DKTFeatures_'+str(config.assignment)+'/val_firstatt_'+str(fold)+'.csv',
                         '../data/DKTFeatures_'+str(config.assignment)+'/test_data.csv', max_step,
                        num_of_questions)


    dtrain = torch.tensor(handle.get_train_data().astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(handle.get_test_data().astype(float).tolist(),
                         dtype=torch.float32)


    train_loader = Data.DataLoader(dtrain, batch_size=config.bs, shuffle=True)
    test_loader = Data.DataLoader(dtest, batch_size=config.bs, shuffle=False)
    return train_loader, test_loader
