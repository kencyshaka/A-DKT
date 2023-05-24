
import os
import random
import wandb
import torch

import torch.optim as optim
import numpy as np

from dataloader import get_data_loader
import evaluation
import warnings
warnings.filterwarnings("ignore")

from c2vRNNModel import c2vRNNModel
from config import Config

wandb.login()
def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    
    config = Config()

    setup_seed(0)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    run = wandb.init(
        # Set the project where this run will be logged
        project="Code-DKT",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": config.lr,
            "epochs": config.epochs,
            "attempts": config.length,
            "batch":config.bs,
            "questions": config.questions,
            "assignment": config.assignment,
            "device": device,
            "code": config.MAX_CODE_LEN*3,
            "question": config.MAX_QUESTION_LEN_partI + config.MAX_QUESTION_LEN_partII,

        })

    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []

    for fold in range(1):
        print("----",fold,"-th run----")

        train_loader, test_loader = get_data_loader(config, config.questions, config.length, fold)
        if config.assignment == 487:
            node_count, path_count = np.load("../data/DKTFeatures_"+str(config.assignment)+"/np_counts_"+str(config.assignment)+"_"+str(fold)+".npy")
        else:
            node_count, path_count = np.load("../data/DKTFeatures_"+str(config.assignment)+"/np_counts.npy")

        model = c2vRNNModel(config,config.model_type, config.questions * 2,
                            config.hidden,
                            config.layers,
                            config.questions,
                            node_count, path_count, device) 

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        loss_func = evaluation.lossFunc(config.questions, config.length, device)
        for epoch in range(config.epochs):
            print('epoch: ' + str(epoch))
            model, optimizer = evaluation.train_epoch(model, train_loader, optimizer,
                                              loss_func, config, device,epoch)
        first_total_scores, first_scores, scores, performance = evaluation.test_epoch(
            model, test_loader, loss_func, device, epoch, config, fold)
        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
    print("Average scores of the first attempts:", np.mean(first_total_scores_list,axis=0))
    print("Average scores of all attempts:", np.mean(performance_list,axis=0))

    #log the table in wandb
    columns = ["category","auc", "f1", "recall", "precision", "acc"]
    table_data = []
    data = np.mean(first_total_scores_list,axis=0)
    table_data.append(["first",data[0], data[1], data[2], data[3], data[4]])
    data = np.mean(performance_list,axis=0)
    table_data.append(["overall", data[0], data[1], data[2], data[3], data[4]])


    data = np.mean(np.array([list(d.values()) for d in first_scores_list]), axis=0)

    for j in range(10):
        table_data.append(["Avg_first problem " + str(j),
                           data[j][0], data[j][1],
                           data[j][2], data[j][3],
                           data[j][4]])

    for i in range(len(first_scores_list)):
        for j in range(10):
            table_data.append(["fold "+str(i)+" problem "+str(j),
                               first_scores_list[i][j][0], first_scores_list[i][j][1],
                               first_scores_list[i][j][2], first_scores_list[i][j][3],
                               first_scores_list[i][j][4]])

            # print(i,j,table_data)

    wandb.log({'Scores for all fold:': wandb.Table(data=table_data, columns=columns)})
    wandb.log({"First_AUC": np.mean(first_total_scores_list,axis=0)[0], "Overall_AUC": np.mean(performance_list,axis=0)[0]})

if __name__ == '__main__':
    main()
