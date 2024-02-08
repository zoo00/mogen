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
device = torch.device("cuda:0" if cuda_condition else "cpu")
    
tokenizer = Tokenizer(gene_vocab,shuf =True)

threshold = None
personalized_genes = False
random_genes = False

Trans_FC = True
Trans_MFC = False
Trans = False
FC = False
MFC = False

if Trans_FC:
    name = 'Trans_FC'
    C_EnC = 'SimpleFC'
    D_EnC = 'SimpleFC'
elif Trans_MFC:
    name = 'Trans_MFC'
    C_EnC = 'MixedFC'
    D_EnC = 'MixedFC'
elif Trans:
    name = 'Trans'
    C_EnC = None
    D_EnC = None
elif MFC:
    name = 'MixedFC'
elif FC:
    name = 'Simple_FC'

nb_epoch=100

gnn_dropout = 0.3
att_dropout = 0.3
fc_dropout = 0.3

nGenes = 300
lr = 0.0001
embed_size = 64
batch_size = 64

heads = 1
layer_drug = 3
dim_drug = 128
nhid = layer_drug*dim_drug
 
att_dim = 512

num_augment = 1

    
title = name+'_Binary_Adim_'+str(att_dim)+'_Ddim_'+str(dim_drug)+'_nGenes_'+str(nGenes)+'_GNN_do'+str(gnn_dropout)+'_att_do_'+str(att_dropout)+'_lr_'+str(lr)
if not(personalized_genes):
    threshold = 4.72

Gene_expression_file = './data/GDSC_micro.BrainArray.RMAlog2Average.ENTREZID.Expr_renamed.tsv'
Drug_info_file = './data/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Drug_feature_file = './data/drug_graph_feat'

drugid2pubchemid, drug_pubchem_id_set, gexpr_feature, drug_feature, experiment_data = get_drug_cell_info(Drug_info_file,Drug_feature_file,
                                                                                                         Gene_expression_file,
                                                                                                         norm = False,threshold = threshold)
if not personalized_genes:
    gexpr_feature = gexpr_feature.T[:300].T
    nGenes = gexpr_feature.shape[1]
    title = 'Fixed_Binary_'+name+'_Adim_'+str(att_dim)+'_Ddim_'+str(dim_drug)+'_nGenes_'+str(nGenes)+'_GNN_do'+str(gnn_dropout)+'_att_do_'+str(att_dropout)+'_lr_'+str(lr)

gexpr_feature.index = gexpr_feature.index.astype(str)
gexpr_feature.columns = gexpr_feature.columns.astype(str)

overlapped_genes = set(gene_vocab['ENTREZID']).intersection(gexpr_feature.columns)    
gexpr_feature = gexpr_feature[overlapped_genes]

over_under_ids_df_list, over_under_genes_df_list = get_binary_gene_set(tokenizer, gexpr_feature, nGenes  = nGenes, 
                                                                      random_genes = True, num_augment = num_augment)

IC = pd.read_csv('./data/binary_IC50.csv')
IC['Drug name'] = IC['Drug name'].astype(str)
IC['Cell line name'] = IC['Cell line name'].astype(str)
data_idx = IC[['Cell line name','Drug name','IC50']].values

drug_dict = np.load('./data/new_drug_feature_graph.npy', allow_pickle=True).item()

input_df_list = [] 

for i in range(num_augment):
    input_df = get_gnn_input_df(data_idx,drug_dict,gexpr_feature,over_under_ids_df_list[i],over_under_genes_df_list[i])
    input_df = input_df[input_df['drug id'] != '84691']
    input_df = input_df[~input_df.duplicated(['cell id','drug id'])]
    if i:
        input_df[(input_df.IC50>0.5)]
    input_df_list.append(input_df)

all_samples =  gexpr_feature.index
  
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
split1 = ss.split(input_df_list[0])
split2 = ss.split(input_df_list[0])


main_fold = 0
for train1_index, test1_index in split1:
    main_fold += 1
    split2 = ss.split(input_df_list[0])
    
    fold = 0 
    for train2_index, test2_index in split2:
        fold += 1
        
        train2_index, val2_index = train_test_split(train2_index, test_size=0.05)
            
        train_df = input_df_list[0].iloc[train2_index]#.reset_index(drop=True)
        val_df = input_df_list[0].iloc[val2_index].reset_index(drop=True)
        test_df = input_df_list[0].iloc[test2_index].reset_index(drop=True)
            
        if num_augment > 1:
            common_index = list(set(input_df_list[1].index.values).intersection(train2_index))
            for k in range(1,num_augment):
                train_df = pd.concat([train_df,input_df_list[k].iloc[common_index]])
            
        train_df = train_df.reset_index(drop=True)
            
        train_dataloader = get_gnn_dataloader(train_df, batch_size=batch_size)
        validation_dataloader = get_gnn_dataloader(val_df, batch_size=batch_size)
        test_dataloader = get_gnn_dataloader(test_df, batch_size=batch_size)
            
        gene_embedding = Gene_Embedding(vocab_size= vocab_size,embed_size=embed_size)
        
        gnn = GNN_drug(layer_drug = layer_drug, dim_drug = dim_drug, do = gnn_dropout)
        if Trans_MFC or Trans_FC or Trans:
            cell_encoder = Transformer_Encoder(genes = nGenes, x_dim= embed_size, y_dim = att_dim, 
                                               dropout = att_dropout, encoder = C_EnC)
            
            drug_encoder = Transformer_Encoder(genes = nGenes, x_dim= nhid, y_dim = att_dim, 
                                               dropout = att_dropout, encoder = D_EnC)
        elif FC:
            cell_encoder = SimpleFC_Encoder(x_dim= embed_size, y_dim = att_dim, dropout = att_dropout)
            
            drug_encoder = SimpleFC_Encoder(x_dim= nhid, y_dim = att_dim, dropout = att_dropout)
            
        elif MFC:
            cell_encoder = MixedFC_Encoder(genes = nGenes, x_dim= embed_size, y_dim = att_dim, 
                                           dropout = att_dropout)
            
            drug_encoder = MixedFC_Encoder(genes = nGenes, x_dim= nhid, y_dim = att_dim, 
                                           dropout = att_dropout)
        
        encoder = Main_Encoder(cell_encoder = cell_encoder, d_dim = nhid, 
                               genes=nGenes, y_dim=att_dim, dropout = att_dropout)
        
        model = GEN(y_dim = att_dim*2, dropout_ratio = fc_dropout,
                 gnn = gnn, embedding = gene_embedding, encoder = encoder)
        
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr) #0.0001
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
            all_y = []
            all_pred_y = []
            sum_loss = 0.0
            count = 0
            model.train()
            for step, (x_drug,x_genes, x_gexpr,y) in enumerate(train_dataloader):
                if len(y) >1:
                    optimizer.zero_grad()
                    x_drug = x_drug.to(device)
                    x_gexpr = x_gexpr.to(device)
                    x_genes = x_genes.to(device)
                    y = y.to(device).float()
                        
                    pred_y = model(x_drug,x_gexpr, x_genes)
                    pred_y = sig(pred_y.view(-1))
                    loss = loss_fun(pred_y,y)
                    sum_loss += loss.item()
                    count +=1
                    loss.backward()
                    optimizer.step()
                    
                    all_y += list(y.cpu().detach().numpy())
                    all_pred_y += list(pred_y.cpu().detach().numpy())
                    
                    
                if (step+1) %500 ==0:
                    print(title)
                    print("training step: ", step)
                    print("step_training loss: ", loss.item())
            
            loss_train = sum_loss/count
            
            AUC, AUPR, F1, ACC, Recall, Specificity, Precision = metrics_graph(all_y,all_pred_y)
            
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
            
            sum_loss = 0.0
            count = 0
            
            all_y = []
            all_pred_y = []
            model.eval()
            for step, (x_drug,x_genes, x_gexpr,y) in enumerate(validation_dataloader):
                if len(y) >1:
                    x_drug = x_drug.to(device)
                    x_gexpr = x_gexpr.to(device)
                    x_genes = x_genes.to(device)
                    y = y.to(device).float()
                        
                    pred_y = model(x_drug,x_gexpr, x_genes)
                    pred_y = sig(pred_y.view(-1))
                    loss = loss_fun(pred_y,y)
                    sum_loss += loss.item()
                    count +=1
                    all_y += list(y.cpu().detach().numpy())
                    all_pred_y += list(pred_y.cpu().detach().numpy())
                    
                    
            loss_val = sum_loss/count
            AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(all_y,all_pred_y)
            
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

            model.eval()
            
            all_y = []
            all_pred_y = []
            sum_loss = 0.0
            count = 0
            
            for step, (x_drug, x_genes, x_gexpr,y) in enumerate(test_dataloader):
                if len(y) >1:
                    x_drug = x_drug.to(device)
                    x_gexpr = x_gexpr.to(device)
                    x_genes = x_genes.to(device)
                    y = y.to(device).float()
                        
                    pred_y = model(x_drug,x_gexpr, x_genes)
                    pred_y = sig(pred_y.view(-1))
                    loss = loss_fun(pred_y,y)
                    sum_loss += loss.item()
                    count +=1
                    
                    all_y += list(y.cpu().detach().numpy())
                    all_pred_y += list(pred_y.cpu().detach().numpy())
                    
                    
            loss_test = sum_loss/count
            AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(all_y,all_pred_y)
            
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
            
            if best_f1 < val_f1[-1]:
                best_f1 = val_f1[-1]
                torch.save(model.state_dict(), save_path+title+'.pt')
                
            if (ep+1) %50 == 0:
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
        all_y = []
        all_pred_y = []
        sum_loss = 0.0
        count = 0
        
        for step, (x_drug, x_genes, x_gexpr,y) in enumerate(test_dataloader):
            if len(y) >1:
                x_drug = x_drug.to(device)
                x_gexpr = x_gexpr.to(device)
                x_genes = x_genes.to(device)
                y = y.to(device).float()
                    
                pred_y = model(x_drug,x_gexpr, x_genes)
                pred_y = sig(pred_y.view(-1))
                loss = loss_fun(pred_y,y)
                sum_loss += loss.item()
                count +=1
                
                all_y += list(y.cpu().detach().numpy())
                all_pred_y += list(pred_y.cpu().detach().numpy())
                
                
        loss_test = sum_loss/count
        AUC, AUPR, F1, ACC,Recall, Specificity, Precision = metrics_graph(all_y,all_pred_y)
        
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
        