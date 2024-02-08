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

gene_vocab = pd.read_csv('./data/compact_gene_vocabulary.csv',sep=',') 
vocab_size = gene_vocab.shape[0]

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:2" if cuda_condition else "cpu")
    
tokenizer = Tokenizer(gene_vocab,shuf =True)

threshold = None
personalized_genes = False
random_genes = False

nb_epoch=250

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

n_cell_layer = 3
n_drug_layer = 2
    
if not(personalized_genes):
    threshold = 4.72

Gene_expression_file = './data/GDSC_micro.BrainArray.RMAlog2Average.ENTREZID.Expr_renamed.tsv'
Drug_info_file = './data/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Drug_feature_file = './data/drug_graph_feat'

drugid2pubchemid, drug_pubchem_id_set, gexpr_feature, _, experiment_data = get_drug_cell_info(Drug_info_file,Drug_feature_file,
                                                                                              Gene_expression_file,
                                                                                              norm = False, threshold = threshold)

gexpr_feature.columns = gexpr_feature.columns.values.astype(str)
overlapped_genes = set(gene_vocab['ENTREZID']).intersection(gexpr_feature.columns)    
gexpr_feature = gexpr_feature[overlapped_genes]

over_under_ids_df, over_under_genes_df = get_gene_set(tokenizer, gexpr_feature, nGenes, random_genes)

data_idx = get_idx(drugid2pubchemid, drug_pubchem_id_set, gexpr_feature,experiment_data)

drug_dict = np.load('./data/new_drug_feature_graph.npy', allow_pickle=True).item()

input_df = get_gnn_input_df(data_idx,drug_dict,gexpr_feature,over_under_ids_df,over_under_genes_df)

input_df = input_df[input_df['drug id'] != '84691']

all_samples =  gexpr_feature.index


import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
n_components = 2

# t-sne 모델 생성
tsne = TSNE(n_components=n_components)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

from sklearn.datasets import load_digits
import seaborn as sns

reg= False
binary = True

test_df = input_df[~input_df.duplicated(['cell id'])]
test_df = test_df.reset_index(drop=True)

#cell_info = pd.read_csv('GDSC_info.csv')
#cell_info['COSMIC identifier'] = cell_info['COSMIC identifier'].astype(str)

#cell_info = cell_info[cell_info['COSMIC identifier'].isin(test_df['cell id'])].sort_values(by=['COSMIC identifier'])

test_df = test_df.sort_values(by=['cell id'])

#test_df['Tissue1'] = cell_info['GDSC Tissue descriptor 1']

#test_df['Tissue2'] = cell_info['GDSC Tissue descriptor 2']
test_df = test_df.reset_index(drop=True)
test_df['label'] = test_df.index
input_genes = torch.tensor(test_df['ids'])
input_scales = torch.tensor(test_df['expres'])


df = pd.DataFrame()
df['TSNE-one'] = np.ones(test_df.shape[0]*5)
df['TSNE-two'] = np.ones(test_df.shape[0]*5)
df['label'] = np.ones(test_df.shape[0]*5)

Fixed = not(personalized_genes)

Trans_FC = False
Trans_MFC = True
Trans = False
FC = False
MFC = False
rotate = [1,0,0,0,0]
for i in range(5):
    Trans_FC = rotate[0]
    Trans_MFC = rotate[1]
    Trans = rotate[2]
    FC = rotate[3]
    MFC = rotate[4]
    last = rotate[len(rotate)-1]
    rotate[i+1:] = rotate[i:len(rotate)-1]
    rotate[0] = last
    rotate[1:i+1] = rotate[:i]
        
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
        name = 'FC'
        
    if Fixed:
        input_type = 'Fixed gene sets'
    else:
        input_type = 'Individual gene sets'
        
    title = input_type+' with '+name
    
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
    
    if reg:
        ## reg fix
        if not(Fixed) and Trans_MFC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_Trans_MFC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and Trans_MFC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_Trans_MFC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if not(Fixed) and Trans_FC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_Trans_FC_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and Trans_FC:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_Trans_FC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
            
        if not(Fixed) and Trans:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_Trans_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and Trans:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_Trans_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
            
        if not(Fixed) and MFC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_MixedFC_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and MFC:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_MixedFC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
                
        if not(Fixed) and FC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_SimpleFC_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and FC:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_Simple_FC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
    
    elif binary: 
        ## binary fix
        if not(Fixed) and Trans_MFC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed3_Trans_MFC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and Trans_MFC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed_Binary_Trans_MFC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if not(Fixed) and Trans_FC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_Trans_FC_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and Trans_FC:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed_Binary_Trans_FC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
            
        if not(Fixed) and Trans:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_Trans_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and Trans:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed_Binary_Trans_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
            
        if not(Fixed) and MFC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_MixedFC_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and MFC:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed_Binary_MixedFC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
                
        if not(Fixed) and FC:
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Only_cell_SimpleFC_Binary_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
        if (Fixed) and FC:   
            model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GEN/weights/Fixed_Binary_Simple_FC_Adim_512_Ddim_128_nGenes_300_GNN_do0.3_att_do_0.3_lr_0.0001_final.pt'))
    
    
    print(title)
    #model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Fixed_Cell_Molecule_E_Adim_512_Ddim_128_nGenes_126_GNN_do0.3_att_do_0.3_lr_7e-05.pt'))
    #model.load_state_dict(torch.load('/NAS_Storage1/leo8544/GE_BERT/CDR_weights/Fixed_Trans2AttDim_512_nGenes_126_GNN_do0.3_att_do_0.3_lr_0.0001_test.pt'))
    
    model.cpu()
    x_g = model.embedding(input_genes, input_scales)
    
    x = model.encoder.cell_encoder(x_g)
    y, _ = torch.max(x, dim= 1)
    
    y = y.cpu().detach().numpy()
    
    tsne_y = tsne.fit_transform(y)
    #tsne_y = pca.fit_transform(y)
    
    test_df['TSNE-one'] = tsne_y[:,0]
    test_df['TSNE-two'] = tsne_y[:,1]
    test_df['label'] = name
    
    if name == 'Trans_FC':
        test_df.index = list(range(0,tsne_y.shape[0]*1))
        
    elif name == 'Trans_MFC':
        test_df.index = list(range(tsne_y.shape[0]*1,tsne_y.shape[0]*2))
        
    elif name == 'MixedFC' :
        test_df.index = list(range(tsne_y.shape[0]*2,tsne_y.shape[0]*3))
    elif name == 'FC' :
        test_df.index = list(range(tsne_y.shape[0]*3,tsne_y.shape[0]*4))
        
    elif name == 'Trans':
        test_df.index = list(range(tsne_y.shape[0]*4,tsne_y.shape[0]*5))
        
    df.update(test_df[['TSNE-one','TSNE-two','label']])
    
    #df.columns = ['TSNE-one','TSNE-two','Method']
    
plt.figure(figsize = (8,6))
p=sns.scatterplot(
        x="TSNE-one", y="TSNE-two",
        hue="label",
        #palette=sns.color_palette("hls", 962),
        data=df,
        legend="full",
        alpha=0.3
    )
p.set_title('Using the same gene set')

