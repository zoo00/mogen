import csv
import pandas as pd
import hickle as hkl
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold

from torch.utils.data import TensorDataset, random_split, DataLoader,Dataset, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt

from torch_geometric.data import Batch   

import torch

import random,os

from sklearn.metrics import roc_auc_score,precision_recall_curve

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MyDataset(Dataset):
    def __init__(self, input_df):
        super(MyDataset, self).__init__()
        self.input_df = input_df
        
    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, index):
        return (self.input_df['drug'][index], self.input_df['ids'][index], self.input_df['expres'][index],self.input_df['IC50'][index])

class MyDataset_simple(Dataset):
    def __init__(self, input_df):
        super(MyDataset_simple, self).__init__()
        self.input_df = input_df
        
    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, index):
        return (self.input_df['drug'][index], self.input_df['expres'][index],self.input_df['IC50'][index])


def _collate(samples):
    drugs, ids, expres, labels = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    return batched_drug, torch.tensor(ids), torch.tensor(expres), torch.tensor(labels)

def collate_simple(samples):
    drugs, expres, labels = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    return batched_drug, torch.tensor(expres), torch.tensor(labels)


def get_gnn_dataloader(df, batch_size=64, simple = False):
    if simple:
        dataset = MyDataset_simple(df)
        collate_fn = collate_simple
    else:
        dataset = MyDataset(df)
        collate_fn = _collate
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    
    return dataloader

def get_gnn_input_df(data_idx,drug_dict,gexpr_feature,over_under_ids_df,over_under_genes_df):
    
    #gexpr_feature.index = gexpr_feature.index.astype(str)
    df = pd.DataFrame(data = data_idx, columns = ['cell id','drug id', 'IC50'])
        
    df['drug'] = 0; 
    #all_drugs = set(df['drug id'])
    for drug_id in drug_dict.keys():
        new_df = pd.DataFrame({'drug': [drug_dict[drug_id]]}, index=df[df['drug id'].isin([str(drug_id)])].index)
        df.update(new_df)
        
    df['ids'] = 0; df['expres'] = 0;
    cell_ids = list(set([i[0] for i in data_idx]))
    #cell_ids = list(set(gexpr_feature.index.values).intersection(cell_ids))
    
    cell_df = pd.DataFrame(index = cell_ids, columns = ['ids','expres'])
    for cell_id in cell_ids:
        genes = over_under_genes_df.loc[cell_id].values
        #gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id][genes].values
        cell_df.loc[cell_id]['ids'] = over_under_ids_df.loc[cell_id].values
        cell_df.loc[cell_id]['expres'] = gexpr_feature.loc[cell_id][genes].values
    
    for cell_id in cell_df.index:
        new_df = pd.DataFrame({'ids': [cell_df['ids'][cell_id]]}, index=df[df['cell id'].isin([cell_id])].index)
        df.update(new_df)
        new_df = pd.DataFrame({'expres': [cell_df['expres'][cell_id]]}, index=df[df['cell id'].isin([cell_id])].index)
        df.update(new_df)
        
    return df

def get_drug_cell_info(Drug_info_file,Drug_feature_file,Gene_expression_file,cancer_response_exp_file,
                       threshold = 6.3, norm = False, small_genes = False):
                   
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[6] for item in rows if item[6].isdigit()}
       

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    

    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])

    if norm:
        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(gexpr_feature.values)
        gexpr_feature = pd.DataFrame(data=scalerGDSC.transform(gexpr_feature.values),
                                     index = gexpr_feature.index,
                                     columns = gexpr_feature.columns)
        
    gexpr_feature.columns = gexpr_feature.columns.astype(str)
    experiment_data = pd.read_csv(cancer_response_exp_file,sep=',',header=0,index_col=[0],engine='python')
    

    for i in range(len(experiment_data.columns)):
        experiment_data.rename(columns={experiment_data.columns[i]:'DATA.'+str(experiment_data.columns[i])},inplace=True)


    
    return drugid2pubchemid, drug_pubchem_id_set, gexpr_feature, drug_feature, experiment_data
    
def get_idx(drugid2pubchemid, drug_pubchem_id_set, gexpr_feature,experiment_data,binary=False):
    
    #filter experiment data
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
     
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in gexpr_feature.index:
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    data_idx.append((each_cellline,pubchem_id,ln_IC50)) 
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'
          %(len(data_idx),nb_celllines,nb_drugs))
        
    return data_idx

def get_input_df(data_idx,drug_feature,gexpr_feature,over_under_ids_df,over_under_genes_df):
    
    #gexpr_feature.index = gexpr_feature.index.astype(str)
    
    df = pd.DataFrame(data = data_idx, columns = ['cell id','drug id', 'IC50'])
        
    
    drug_df = pd.DataFrame(index = drug_feature.keys(),columns = ['data'])
    for drug_id in drug_feature.keys():
        feat_mat,adj_list,_ = drug_feature[drug_id]
        drug_df.loc[drug_id]['data'] = CalculateGraphFeat(feat_mat,adj_list)
    
    df['drug1'] = 0; df['drug2'] = 0; 
    
    for drug_id in drug_df.index:
        new_df = pd.DataFrame({'drug1': [drug_df['data'][drug_id][0]]}, index=df[df['drug id'].isin([drug_id])].index)
        df.update(new_df)
        new_df = pd.DataFrame({'drug2': [drug_df['data'][drug_id][1]]}, index=df[df['drug id'].isin([drug_id])].index)
        df.update(new_df)
        
    df['ids'] = 0; df['expres'] = 0;
    cell_ids = list(set([i[0] for i in data_idx]))
    #cell_ids = list(set(gexpr_feature.index.values).intersection(cell_ids))
    
    cell_df = pd.DataFrame(index = cell_ids, columns = ['ids','expres'])
    for cell_id in cell_ids:
        genes = over_under_genes_df.loc[cell_id].values
        #gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id][genes].values
        cell_df.loc[cell_id]['ids'] = over_under_ids_df.loc[cell_id].values
        cell_df.loc[cell_id]['expres'] = gexpr_feature.loc[cell_id][genes].values
    
    for cell_id in cell_df.index:
        new_df = pd.DataFrame({'ids': [cell_df['ids'][cell_id]]}, index=df[df['cell id'].isin([cell_id])].index)
        df.update(new_df)
        new_df = pd.DataFrame({'expres': [cell_df['expres'][cell_id]]}, index=df[df['cell id'].isin([cell_id])].index)
        df.update(new_df)
        
    return df

def get_simple_input_df(data_idx,drug_feature,gexpr_feature,drug_dict = {}):
    
    #gexpr_feature.index = gexpr_feature.index.astype(str)
    
    df = pd.DataFrame(data = data_idx, columns = ['cell id','drug id', 'IC50'])
        
    if len(drug_dict):
        df['drug'] = 0; 
        
        for drug_id in drug_dict.keys():
            new_df = pd.DataFrame({'drug': [drug_dict[drug_id]]}, index=df[df['drug id'].isin([str(drug_id)])].index)
            df.update(new_df)
    else:
        drug_df = pd.DataFrame(index = drug_feature.keys(),columns = ['data'])
        
        for drug_id in drug_feature.keys():
            feat_mat,adj_list,_ = drug_feature[drug_id]
            drug_df.loc[drug_id]['data'] = CalculateGraphFeat(feat_mat,adj_list)
        
        df['drug1'] = 0; df['drug2'] = 0; 
        
        for drug_id in drug_df.index:
            new_df = pd.DataFrame({'drug1': [drug_df['data'][drug_id][0]]}, index=df[df['drug id'].isin([drug_id])].index)
            df.update(new_df)
            new_df = pd.DataFrame({'drug2': [drug_df['data'][drug_id][1]]}, index=df[df['drug id'].isin([drug_id])].index)
            df.update(new_df)
        
    df['expres'] = 0;
    cell_ids = list(set([i[0] for i in data_idx]))
    cell_df = pd.DataFrame(index = cell_ids, columns = ['expres'])
    for cell_id in cell_ids:
        cell_df.loc[cell_id]['expres'] = gexpr_feature.loc[cell_id].values
    
    for cell_id in cell_df.index:
        new_df = pd.DataFrame({'expres': [cell_df['expres'][cell_id]]}, index=df[df['cell id'].isin([cell_id])].index)
        df.update(new_df)
        
    return df

def get_dataloader(drug_ids,sample_ids,input_df,simple = False, batch_size=64, val_ratio=None, test_ratio = None):
    df = input_df
    if sample_ids:
        df = df[df['cell id'].isin(sample_ids)]
    if drug_ids:
        df = df[df['drug id'].isin(drug_ids)]
    
    X_drug_feat_data = [item for item in df['drug1']]
    X_drug_adj_data = [item for item in df['drug2']]
    X_drug_feat_data = np.array(X_drug_feat_data)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data = np.array(X_drug_adj_data)#nb_instance * Max_stom * Max_stom
    
    X_drug_feat_data = torch.Tensor(X_drug_feat_data)
    X_drug_adj_data = torch.Tensor(X_drug_adj_data)
    
    
    X_gexpr_data = [item for item in df['expres']]
    X_gexpr_data = torch.Tensor(X_gexpr_data)
    
    Y = [item for item in df['IC50']]
    Y = torch.Tensor(Y)
    
    if not simple:
        X_genes_data = [item for item in df['ids']]
        X_genes_data = torch.Tensor(X_genes_data)
        X_genes_data = X_genes_data.type(torch.int64)
        
        dataset = TensorDataset(X_drug_feat_data,X_drug_adj_data, X_genes_data,X_gexpr_data,Y)
    else:
        dataset = TensorDataset(X_drug_feat_data,X_drug_adj_data, X_gexpr_data,Y)
    if val_ratio:
        
        if test_ratio:
            test_size = int(test_ratio * len(dataset))
            train_size = len(dataset) - test_size
            dataset, test_dataset = random_split(dataset, [train_size, test_size])
            test_dataloader = DataLoader(
                        test_dataset, # The validation samples.
                        sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                        batch_size = batch_size # Evaluate with this batch size.
                    )
            
        val_size = int(val_ratio * len(dataset))
        train_size = len(dataset) - val_size
            
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        
        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        
    else:
        train_dataloader = DataLoader(
                    dataset,  # The training samples.
                    sampler = RandomSampler(dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        
    if test_ratio:
        return train_dataloader, validation_dataloader, test_dataloader
    elif val_ratio:
        return train_dataloader, validation_dataloader
    else:
        return train_dataloader

def show_picture(train,val, test, title, path='' ,save=False):
    plt.plot(train)
    plt.plot(val)
    plt.plot(test)
    plt.legend(['train','val','test'])
    plt.title(title)
    if save:
        plt.savefig(path+title+'.png')
    
    plt.show()
    
def get_rand_genes(tokenizer, x_samples, nGenes, gexpr_feature):
    gexpr = gexpr_feature.loc[x_samples]
    genes = gexpr_feature.columns.values
    rand_genes = [random.choices(population=genes, k=nGenes) for i in range(gexpr.shape[0])]
    rand_ids = torch.tensor([tokenizer.convert_symb_to_id(genes) for genes in rand_genes])
    rand_exprs  = torch.tensor([gexpr.iloc[i][rand_genes[i]].values for i in range(len(rand_genes))])
    
    return rand_ids,rand_exprs

Max_atoms = 100

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if True:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]


def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1,acc,recall, specificity, precision
    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0], recall[0, 0], specificity[0, 0], precision[0, 0]

def get_binary_gene_set(tokenizer, gexpr_feature, nGenes, random_genes = True, num_augment = 1):
    abs_gexpr = np.abs(gexpr_feature-np.mean(gexpr_feature))
    
    all_genes = list(gexpr_feature.columns)
    
    over_under_genes_list = [[] for i in range(num_augment)]
    over_under_ids_list = [[] for i in range(num_augment)]
    
    for i in range(abs_gexpr.shape[0]):
        
        for j in range(num_augment):
            if j ==0 or not(random_genes):
                genes = abs_gexpr.iloc[i].sort_values(ascending = False)[nGenes*j:nGenes*(j+1)].index
            else:
                genes = random.sample(all_genes, nGenes)
                
            ids = tokenizer.convert_symb_to_id(genes)
            
            over_under_ids_list[j].append(ids)
            over_under_genes_list[j].append(genes)
    
    over_under_ids_df_list = [pd.DataFrame(data = over_under_ids_list[i],index = gexpr_feature.index) for i in range(num_augment)]
    over_under_genes_df_list = [pd.DataFrame(data = over_under_genes_list[i],index = gexpr_feature.index) for i in range(num_augment)]
    return over_under_ids_df_list, over_under_genes_df_list


def get_gene_set(tokenizer, gexpr_feature, nGenes, random_genes = True):
    abs_gexpr = np.abs(gexpr_feature-np.mean(gexpr_feature))
    over_under_genes = []
    over_under_ids = []
    
    #all_genes = list(gexpr_feature.columns)
    
    for i in range(abs_gexpr.shape[0]):
        #genes = random.sample(all_genes, nGenes)
        genes = abs_gexpr.iloc[i].sort_values(ascending = False)[:nGenes].index
        ids = tokenizer.convert_symb_to_id(genes)
        
        over_under_ids.append(ids)
        over_under_genes.append(genes)
    
    over_under_ids_df = pd.DataFrame(data = over_under_ids,index = gexpr_feature.index)
    over_under_genes_df = pd.DataFrame(data = over_under_genes,index = gexpr_feature.index)
    
    return  over_under_ids_df,over_under_genes_df