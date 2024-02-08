import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch

from torch import optim

from utils import *
from models import *

gene_vocab = pd.read_csv('./data/compact_gene_vocabulary.csv',sep=',') 
vocab_size = gene_vocab.shape[0]

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:3" if cuda_condition else "cpu")
    
tokenizer = Tokenizer(gene_vocab,shuf =True)

use_DeepCDR = False
use_GEN = True

threshold = None
personalized_genes = True
random_genes = False

nb_epoch=100
lr = 0.0001

gcn_dropout = 0.3
att_dropout = 0.3
fc_dropout = 0.3

heads = 1
layer_drug = 3
dim_drug = 128
nhid = layer_drug*dim_drug
 
att_dim = 512

#threshold = 6.3 #genes 126
#threshold = 8.1 # genes 64
threshold = 4.72

batch_size = 64

    
Gene_expression_file = './data/GDSC_micro.BrainArray.RMAlog2Average.ENTREZID.Expr_renamed.tsv'
Drug_info_file = './data/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Drug_feature_file = './data/drug_graph_feat'
drugid2pubchemid, drug_pubchem_id_set, gexpr_feature, drug_feature, experiment_data = get_drug_cell_info(Drug_info_file,Drug_feature_file,
                                                                                                         Gene_expression_file,
                                                                                                         norm = False,threshold = threshold)

if threshold == 4.72:
    gexpr_feature = gexpr_feature.T[:300].T


IC = pd.read_csv('./data/binary_IC50.csv')
IC['Drug name'] = IC['Drug name'].astype(str)
IC['Cell line name'] = IC['Cell line name'].astype(int)
data_idx = IC[['Cell line name','Drug name','IC50']].values


drug_dict = np.load('./data/new_drug_feature_graph.npy', allow_pickle=True).item()

input_df = get_simple_input_df(data_idx,drug_feature,gexpr_feature, drug_dict = drug_dict)

input_df = input_df[input_df['drug id'] != '84691']
input_df = input_df.reset_index(drop=True)

all_samples =  gexpr_feature.index

nGenes = gexpr_feature.shape[1]

if use_DeepCDR:
    title = 'DeepCDR_GIN_Binary_nGenes_'+str(nGenes)
elif use_GEN:
    title = 'GEN_WO_GeneVec_Binary_Adim_'+str(att_dim)+'_Ddim_'+str(dim_drug)+'_nGenes_'+str(nGenes)+'_GNN_do'+str(gcn_dropout)+'_att_do_'+str(att_dropout)+'_lr_'+str(lr)

save_path = './weights/'
img_path = './imgs/'
result_path = './results/'
        
total_train_auc = []
total_val_auc = []
total_test_auc = []

total_train_aupr = []
total_val_aupr = []
total_test_aupr = []

total_train_f1 = []
total_val_f1 = []
total_test_f1 = []

total_train_acc = []
total_val_acc = []
total_test_acc = []

total_train_recall = []
total_val_recall = []
total_test_recall = []

total_train_specificity = []
total_val_specificity = []
total_test_specificity = []

total_train_precision = []
total_val_precision = []
total_test_precision = []

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
        
        layer_drug = 3
        dim_drug = 128
        nhid = layer_drug*dim_drug
        
        gcn = GNN_drug(layer_drug = layer_drug, dim_drug = dim_drug, do=gcn_dropout)
        
        if use_DeepCDR:
            model = DeepCDR_GIN(gcn = gcn, gexpr_dim=nGenes, dropout =fc_dropout, gnn_dim = nhid)
            
        elif use_GEN:
            model = GEN_WO_GeneVec(gcn = gcn, gexpr_dim=nGenes, dropout_drug = gcn_dropout,
                                   dropout_cell = att_dropout, dropout_reg = fc_dropout,
                                   d_dim = nhid, y_dim = att_dim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr= lr)
        loss_fun = nn.BCELoss()
        sig = nn.Sigmoid()
        
        train_auc = []
        test_auc = []
        val_auc = []
        
        train_aupr = []
        test_aupr = []
        val_aupr = []
        
        train_f1 = []
        test_f1 = []
        val_f1 = []
        
        train_acc = []
        test_acc = []
        val_acc = []
        
        train_recall = []
        val_recall = []
        test_recall = []
        
        train_specificity = []
        val_specificity = []
        test_specificity = []
        
        train_precision = []
        val_precision = []
        test_precision = []
        
        train_loss = []
        test_loss = []
        val_loss = []
        
        best_f1 = 0.
        
        for ep in range(nb_epoch):
            sum_loss = 0.
            count = 0
            
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
                    pred_y = sig(pred_y.view(-1))
                    loss = loss_fun(pred_y,y)
                    loss.backward()
                    optimizer.step()
                    
                    pred_y = pred_y.detach().cpu().numpy()
                    
                    y = y.detach().cpu().numpy()
                    
                    true_Y += list(y)
                    pred_Y += list(pred_y)
                    
                    sum_loss += loss.item()
                    count +=1
                    
                    if (step+1) %500 ==0:
                        print("training step: ", step)
                        print("step_training loss: ", loss.item())
                        
            loss_train = sum_loss/count
            AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(true_Y,pred_Y)
            
            print("Train avg_loss: ", loss_train)
            print("Train AUC: ", AUC)
            print("Train AUPR: ", AUPR)
            print("Train F1: ", F1)
            print("Train ACC: ", ACC)
            print("Train Recall: ", Recall)
            print("Train Specificity: ", Specificity)
            print("Train Precision: ", Precision)
            
            train_auc.append(AUC)
            train_aupr.append(AUPR)
            train_f1.append(F1)
            train_acc.append(ACC)
            train_recall.append(Recall)
            train_specificity.append(Specificity)
            train_precision.append(Precision)

            train_loss.append(loss_train)
            
            sum_val_loss = 0.
            count = 0
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
                    pred_y = sig(pred_y.view(-1))
                    loss = loss_fun(pred_y,y)
                    
                    pred_y = pred_y.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    
                    true_Y += list(y)
                    pred_Y += list(pred_y)
                    
                    sum_val_loss += loss.item()
                    count +=1
            
            loss_val = sum_val_loss/count
            AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(true_Y,pred_Y)
            print("Validation avg_loss: ", loss_val)
            print("Validation AUC: ", AUC)
            print("Validation AUPR: ", AUPR)
            print("Validation F1: ", F1)
            print("Validation ACC: ", ACC)
            print("Validation Recall: ", Recall)
            print("Validation Specificity: ", Specificity)
            print("Validation Precision: ", Precision)
            
            val_loss.append(loss_val)
            
            val_auc.append(AUC)
            val_aupr.append(AUPR)
            val_f1.append(F1)
            val_acc.append(ACC)
            val_recall.append(Recall)
            val_specificity.append(Specificity)
            val_precision.append(Precision)
            
            if best_f1 < val_f1[-1]:
                best_f1 = val_f1[-1]
                torch.save(model.state_dict(), save_path+title+'.pt')
                
            true_Y = []
            pred_Y = []
            
            sum_test_loss = 0.
            count = 0
            
            model.eval()
            for step, (x_drug, x_gexpr,y) in enumerate(test_dataloader):
                if len(y) >1:
                    x_gexpr = x_gexpr.to(device).float()
                    x_drug = x_drug.to(device)
                    y = y.to(device).float()
                    
                    pred_y = model(x_feat =  x_drug, x_gexpr = x_gexpr)
                    pred_y = sig(pred_y.view(-1))
                    loss = loss_fun(pred_y,y)
                    
                    pred_y = pred_y.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    true_Y += list(y)
                    pred_Y += list(pred_y)
                    
                    sum_test_loss += loss.item()   
                    count +=1
                    
            
            loss_test = sum_test_loss/count
            AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(true_Y,pred_Y)
            print("Test avg_loss: ", loss_test)
            print("Test AUC: ", AUC)
            print("Test AUPR: ", AUPR)
            print("Test F1: ", F1)
            print("Test ACC: ", ACC)
            print("Test Recall: ", Recall)
            print("Test Specificity: ", Specificity)
            print("Test Precision: ", Precision)
            
            test_auc.append(AUC)
            test_aupr.append(AUPR)
            test_f1.append(F1)
            test_acc.append(ACC)
            test_recall.append(Recall)
            test_specificity.append(Specificity)
            test_precision.append(Precision)

            test_loss.append(loss_test)
            
            if (ep+1) %50 ==0:
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Loss_'+title
                show_picture(train_loss,val_loss, test_loss, input_title)
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_AUC_'+title
                show_picture(train_auc,val_auc, test_auc, input_title)
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_AUPR_'+title
                show_picture(train_aupr,val_aupr, test_aupr, input_title)
                input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_F1_'+title
                show_picture(train_f1,val_f1, test_f1, input_title)
                
            print("#################### epoch ############################ ",ep)
            
        model.load_state_dict(torch.load(save_path+title+'.pt'))
        torch.save(model.state_dict(), save_path+title+'_final.pt')
        
        true_Y = []
        pred_Y = []
        
        sum_test_loss = 0.
        count = 0
        
        model.eval()
        for step, (x_drug, x_gexpr,y) in enumerate(test_dataloader):
            if len(y) >1:
                x_gexpr = x_gexpr.to(device).float()
                x_drug = x_drug.to(device)
                y = y.to(device).float()
                
                pred_y = model(x_feat =  x_drug, x_gexpr = x_gexpr)
                pred_y = sig(pred_y.view(-1))
                loss = loss_fun(pred_y,y)
                
                pred_y = pred_y.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                true_Y += list(y)
                pred_Y += list(pred_y)
                
                sum_test_loss += loss.item()   
                count +=1
        
        loss_test = sum_test_loss/count
        AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(true_Y,pred_Y)
        
        print("Test avg_loss: ", loss_test)
        print("Test AUC: ", AUC)
        print("Test AUPR: ", AUPR)
        print("Test F1: ", F1)
        print("Test ACC: ", ACC)
        print("Test Recall: ", Recall)
        print("Test Specificity: ", Specificity)
        print("Test Precision: ", Precision)
        
        test_auc.append(AUC)
        test_aupr.append(AUPR)
        test_f1.append(F1)
        test_acc.append(ACC)
        test_recall.append(Recall)
        test_specificity.append(Specificity)
        test_precision.append(Precision)
        test_loss.append(loss_test)
        
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Loss_'+title
        show_picture(train_loss,val_loss, test_loss, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_AUC_'+title
        show_picture(train_auc,val_auc, test_auc, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_AUPR_'+title
        show_picture(train_aupr,val_aupr, test_aupr, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_F1_'+title
        show_picture(train_f1,val_f1, test_f1, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_ACC_'+title
        show_picture(train_acc,val_acc, test_acc, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Recall_'+title
        show_picture(train_recall,val_recall, test_recall, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Specificity_'+title
        show_picture(train_specificity,val_specificity, test_specificity, input_title,path=img_path, save=True)
        input_title = 'Fold_'+str(main_fold)+'X'+str(fold)+'_Precision_'+title
        show_picture(train_precision,val_precision, test_precision, input_title,path=img_path, save=True)
        
        total_test_precision.append(Precision)
        total_test_specificity.append(Specificity)
        total_test_acc.append(ACC)
        total_test_f1.append(F1)
        total_test_aupr.append(AUPR)
        total_test_auc.append(AUC)
        total_test_recall.append(Recall)
        total_test_losses.append(loss_test)

        df_test_auc = pd.DataFrame(data = total_test_auc)
        df_test_aupr = pd.DataFrame(data = total_test_aupr)
        df_test_f1 = pd.DataFrame(data = total_test_f1)
        df_test_acc = pd.DataFrame(data = total_test_acc)
        df_test_losses = pd.DataFrame(data = total_test_losses)
        df_test_recall = pd.DataFrame(data = total_test_recall)
        df_test_specificity = pd.DataFrame(data = total_test_specificity)
        df_test_precision = pd.DataFrame(data = total_test_precision)

        df_test_auc.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_AUC.csv')
        df_test_aupr.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_AUPR.csv')
        df_test_f1.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_F1.csv')
        df_test_acc.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_ACC.csv')
        df_test_recall.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_Recall.csv')
        df_test_specificity.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_Specificity.csv')
        df_test_precision.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_Precision.csv')
        df_test_losses.to_csv(result_path+'Fold_'+str(main_fold)+'_'+title+'_loss.csv')
    
