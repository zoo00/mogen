import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy.stats import pearsonr

import torch.nn as nn
import torch

from torch import optim
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error
from utils import *
from models import *

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda_condition else "cpu")

use_DeepCDR = True
use_GEN = False

nb_epoch=250
lr = 0.0001

gcn_dropout = 0.3
att_dropout = 0.3
fc_dropout = 0.3

heads = 1
layer_drug = 3
dim_drug = 128
nhid = layer_drug*dim_drug

att_dim = 512

#threshold = 8.1 # genes 64
#threshold = 6.3

threshold = 4.72
batch_size = 64

#dataset = 'GDSC'
#dataset = 'CCLE'
dataset = 'CTRP'

if dataset == 'GDSC':
    threshold = 4.72
    Gene_expression_file = './data/GDSC_micro.BrainArray.RMAlog2Average.ENTREZID.Expr_renamed.tsv'
    Drug_info_file = './data/1.Drug_listMon Jun 24 09_00_55 2019.csv'
    Drug_feature_file = './data/drug_graph_feat'
    cancer_response_exp_file = './data/GDSC_IC50.csv'

    drugid2pubchemid, drug_pubchem_id_set, gexpr_feature, drug_feature, experiment_data = get_drug_cell_info(Drug_info_file,Drug_feature_file,
                                                                                                         Gene_expression_file, cancer_response_exp_file,
                                                                                                         norm = False,threshold = threshold)

    if threshold == 4.72:
        gexpr_feature = gexpr_feature.T[:300].T

    data_idx = get_idx(drugid2pubchemid, drug_pubchem_id_set, gexpr_feature,experiment_data)
    drug_dict = np.load('./data/new_drug_feature_graph.npy', allow_pickle=True).item()
    input_df = get_simple_input_df(data_idx,drug_feature,gexpr_feature, drug_dict = drug_dict)
    input_df = input_df[input_df['drug id'] != '84691']
    input_df = input_df.reset_index(drop=True)


else:
    if dataset == 'CCLE':
        nb_epoch= 50
        threshold = 0.031
        drug_dict_file = './data/ccle_graph.npy'
        gexpr_feature = pd.read_csv('./data/CCLE_RNAseq.csv',index_col = 0)
        data_idx = pd.read_csv('./data/CCLE_cell_drug_labels.csv',index_col=0).values

    elif dataset == 'CTRP':
        threshold = 3.93
        drug_dict_file = './data/ctrp_graph.npy'

        gexpr_feature = pd.read_csv('./data/data_CTRP-Broad-MIT_exp.txt',sep='\t',index_col = 0).T
        data_idx = pd.read_csv('./data/CTRP_y.csv').values


    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold) #CCLE threshold == 3.9
    selector.fit_transform(gexpr_feature).shape

    gexpr_feature = gexpr_feature[gexpr_feature.columns[selector.get_support(indices=True)]]

    drug_dict = np.load(drug_dict_file, allow_pickle=True).item()

    input_df = get_simple_input_df(data_idx,None,gexpr_feature, drug_dict = drug_dict)

all_samples =  gexpr_feature.index

nGenes = gexpr_feature.shape[1]

if use_DeepCDR:
    title = 'DeepCDR_GIN2_nGenes_'+str(nGenes)
elif use_GEN:
    title = 'GEN_WO_GeneVec_Adim_'+str(att_dim)+'_Ddim_'+str(dim_drug)+'_nGenes_'+str(nGenes)+'_GNN_do'+str(gcn_dropout)+'_att_do_'+str(att_dropout)+'_lr_'+str(lr)

save_path = './weights/'
img_path = './imgs/'
result_path = './results/'

total_train_pcc = []
total_val_pcc = []
total_test_pcc = []
total_train_r2 = []
total_val_r2 = []
total_test_r2 = []
total_train_losses = []
total_test_losses = []
total_val_losses = []


from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2)

split1 = ss.split(input_df)
split2 = ss.split(input_df)

main_fold = 0
for train1_index, test1_index in split1:
    main_fold += 1

    split2 = ss.split(input_df)

    fold = 0
    for train2_index, test2_index in split2:
        fold += 1

        train2_index, val2_index = train_test_split(train2_index, test_size=0.05)
        train_df = input_df.iloc[train2_index].reset_index(drop=True)
        val_df = input_df.iloc[val2_index].reset_index(drop=True)
        test_df = input_df.iloc[test2_index].reset_index(drop=True)

        train_dataloader = get_gnn_dataloader(train_df, batch_size=batch_size,simple = True)
        validation_dataloader = get_gnn_dataloader(val_df, batch_size=batch_size,simple = True)
        test_dataloader = get_gnn_dataloader(test_df, batch_size=batch_size,simple = True)

        gcn = GNN_drug(layer_drug = layer_drug, dim_drug = dim_drug, do=gcn_dropout)

        if use_DeepCDR:
            model = DeepCDR_GIN(gcn = gcn, gexpr_dim=nGenes, dropout =fc_dropout, gnn_dim = nhid)

        elif use_GEN:
            model = GEN_WO_GeneVec(gcn = gcn, gexpr_dim=nGenes, dropout_drug = gcn_dropout,
                                   dropout_cell = att_dropout, dropout_reg = fc_dropout,
                                   d_dim = nhid, y_dim = att_dim)

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr= lr)
        mse = nn.MSELoss()

        train_pcc = []
        val_pcc = []
        test_pcc = []

        train_r2 = []
        val_r2 = []
        test_r2 = []

        best_r2 = 0
        train_loss = []
        test_loss = []
        val_loss = []

        for ep in range(nb_epoch):

            true_Y = []
            pred_Y = []

            model.train()
            for step, (x_drug, x_gexpr,y) in enumerate(train_dataloader):
                if len(y) >1:
                    optimizer.zero_grad()
                    x_gexpr = x_gexpr.to(device).float()
                    x_drug = x_drug.to(device)
                    y = y.to(device).float()

                    pred_y = model(x_feat =  x_drug, x_gexpr = x_gexpr)
                    loss = mse(pred_y.view(-1),y)
                    loss.backward()
                    optimizer.step()

                    pred_y = pred_y.view(-1).detach().cpu().numpy()
                    y = y.detach().cpu().numpy()

                    true_Y += list(y)
                    pred_Y += list(pred_y)

                    overall_pcc = pearsonr(pred_y,y)[0]
                    if (step+1) %500 ==0:
                        print(title)
                        print("training step: ", step)
                        print("step_training loss: ", loss.item())
                        print("The overall Pearson's correlation is %.4f."%overall_pcc)

            loss_train = mean_squared_error(true_Y, pred_Y)
            pcc_train = pearsonr(true_Y, pred_Y)[0]
            r2_train = r2_score(true_Y, pred_Y)
            print("Train avg_loss: ", loss_train)
            print("Train avg_pcc: ", pcc_train)
            print("Train r2: ", r2_train)

            train_pcc.append(pcc_train)
            train_loss.append(loss_train)
            train_r2.append(r2_train)

            true_Y = []
            pred_Y = []

            model.eval()
            for step, (x_drug, x_gexpr,y) in enumerate(validation_dataloader):
                if len(y) >1:
                    optimizer.zero_grad()
                    x_gexpr = x_gexpr.to(device).float()
                    x_drug = x_drug.to(device)
                    y = y.to(device).float()

                    pred_y = model(x_feat =  x_drug, x_gexpr = x_gexpr)

                    loss = mse(pred_y.view(-1),y)

                    pred_y = pred_y.view(-1).detach().cpu().numpy()
                    y = y.detach().cpu().numpy()

                    true_Y += list(y)
                    pred_Y += list(pred_y)


            loss_val = mean_squared_error(true_Y,pred_Y)
            pcc_val = pearsonr(true_Y,pred_Y)[0]
            r2_val = r2_score(true_Y,pred_Y)

            print("Validation avg_loss: ", loss_val)
            print("Validation avg_pcc: ", pcc_val)
            print("Validation r2: ", r2_val)
            val_loss.append(loss_val)
            val_pcc.append(pcc_val)
            val_r2.append(r2_val)

            if best_r2 < val_r2[-1]:
                best_r2 = val_r2[-1]
                torch.save(model.state_dict(), save_path+title+'.pt')
                print('Best r2', best_r2)

            true_Y = []
            pred_Y = []

            model.eval()
            for step, (x_drug, x_gexpr,y) in enumerate(test_dataloader):
                if len(y) >1:
                    x_gexpr = x_gexpr.to(device).float()
                    x_drug = x_drug.to(device)
                    y = y.to(device).float()

                    pred_y = model(x_feat =  x_drug, x_gexpr = x_gexpr)

                    loss = mse(pred_y.view(-1),y)

                    pred_y = pred_y.view(-1).detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    true_Y += list(y)
                    pred_Y += list(pred_y)


            loss_test = mean_squared_error(true_Y,pred_Y)
            pcc_test = pearsonr(true_Y,pred_Y)[0]
            r2_test = r2_score(true_Y,pred_Y)

            print("Test avg_loss: ", loss_test)
            print("Test avg_pcc: ", pcc_test)
            print("Test r2: ", r2_test)

            test_pcc.append(pcc_test)
            test_loss.append(loss_test)
            test_r2.append(r2_test)

            if (ep+1) %50 ==0:
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Loss_'+title
                show_picture(train_loss,val_loss, test_loss, input_title)
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_PCC_'+title
                show_picture(train_pcc,val_pcc, test_pcc, input_title)
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_r2_'+title
                show_picture(train_r2,val_r2, test_r2, input_title)


            print("#################### epoch ############################ ",ep)

        model.load_state_dict(torch.load(save_path+title+'.pt'))
        torch.save(model.state_dict(), save_path+title+'_final.pt')
        all_y = []
        all_pred_y = []
        sum_loss = 0.0
        count = 0

        model.eval()
        for step, (x_drug, x_gexpr,y) in enumerate(test_dataloader):
            if len(y) >1:
                x_gexpr = x_gexpr.to(device).float()
                x_drug = x_drug.to(device)
                y = y.to(device).float()

                pred_y = model(x_feat =  x_drug, x_gexpr = x_gexpr)

                loss = mse(pred_y.view(-1),y)

                pred_y = pred_y.view(-1).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                true_Y += list(y)
                pred_Y += list(pred_y)

        loss_test = mean_squared_error(true_Y,pred_Y)
        pcc_test = pearsonr(true_Y,pred_Y)[0]
        r2_test = r2_score(true_Y,pred_Y)

        print("Test avg_loss: ", loss_test)
        print("Test avg_pcc: ", pcc_test)
        print("Test r2: ", r2_test)

        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Loss_'+title
        show_picture(train_loss,val_loss, test_loss, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_PCC_'+title
        show_picture(train_pcc,val_pcc, test_pcc, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_r2_'+title
        show_picture(train_r2,val_r2, test_r2, input_title,path=img_path, save=True)

        total_train_pcc.append(train_pcc)
        total_val_pcc.append(val_pcc)
        total_test_pcc.append(pcc_test)

        total_train_r2.append(train_r2)
        total_val_r2.append(val_r2)
        total_test_r2.append(r2_test)

        total_train_losses.append(train_loss)
        total_val_losses.append(val_loss)
        total_test_losses.append(loss_test)

        df_test_pcc = pd.DataFrame(data = total_test_pcc)
        df_test_r2 = pd.DataFrame(data = total_test_r2)

        df_test_losses = pd.DataFrame(data = total_test_losses)

        df_test_pcc.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_pcc.csv')
        df_test_r2.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_r2.csv')

        df_test_losses.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_loss.csv')
