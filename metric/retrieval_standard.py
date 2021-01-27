import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm


index = 7
name = 'mv_pn_mi_2'
path = ['sketch_feat.npy','model_feat.npy' ]

p_q = path[0] 
p_c = path[1] 

def get_data(p_q, p_c):
    #a = loadmat(p_q)
    #b = loadmat(p_c)
    a = np.load(p_q,allow_pickle=True).item()
    b = np.load(p_c,allow_pickle=True).item()
    #print('a keys:', a.keys())
    lens = -1 
    fts_q = a['fts']  # feature qurey
    las_q = a['las']  # label
    fts_c = b['fts']  # feature contrain
    las_c = b['las']  # label
    #print(fts_q.shape, fts_c.shape)
    las_q = np.array(las_q).reshape(-1)  # 规范标签的形状
    las_c = np.array(las_c).reshape(-1)
    return fts_q, las_q, fts_c, las_c

# 输入q,c batch x feature
# 输出：q * c  --query x contain
def dist_euler(fts_q, fts_c):
    fts_qs = np.sum(np.square(fts_q),axis=-1,keepdims=True)
    fts_cs = np.sum(np.square(fts_c),axis=-1,keepdims=True).T
    qc = np.matmul(fts_q,fts_c.T)
    dist = fts_qs + fts_cs - 2 * qc
    return dist

def dist_cos(fts_q, fts_c):
    
    up = np.matmul(fts_q,fts_c.T)
    down1 = np.sqrt(np.sum(np.square(fts_q),axis=-1,keepdims=True))
    down2  = np.sqrt(np.sum(np.square(fts_c),axis=-1,keepdims=True).T)
    down = np.matmul(down1, down2)
    dist = up/(down+1e-4)
    return 1-dist

def get_pr_cls(dist_p=0, data=0):
    #print('----',fts_q.shape, fts_c.shape)
    fts_q, las_q, fts_c, las_c = data
    if dist_p==0:
        dist = dist_euler(fts_q, fts_c)
    else : dist = dist_cos(fts_q, fts_c)
    
    dist_index = np.argsort(dist, axis=-1)
    dist = np.sort(dist, axis=-1)


    # 利用标签计算，标记检索结果
    len_q,len_c = dist.shape
    
    cls_num = np.sort(np.unique(las_c))
    num_per_cls = np.array([ np.sum(las_c==i) for i in cls_num])  # 目标域各类的数目
    C = num_per_cls[las_q]
    
    result = np.zeros_like(dist)
    laq_bool = np.tile(las_q,(len_c,1)).T
    lac_bool = np.tile(las_c,(len_q,1))  # 需要对c 的标签进一步排序
    index = np.tile(np.array(range(len_q)),(len_c,1)).T
    lac_bool = lac_bool[index,dist_index]
    result = (laq_bool == lac_bool)

    p = np.zeros(len_c)
    r = np.zeros(len_c)
    r_all = np.sum(result)
    for i in range(len_c):
        s = np.sum(result[:,:i+1])
        p[i] = s/((i+1)*len_q)
        r[i] = s/r_all
    mAP = np.sum((r[1 :] - r[:-1])*(p[:-1]+p[1 :]))/2
    mAP5 = p[:5].mean()
    # NN
    NN = result[:,0].sum()/(result.shape[0]);

    # FT,ST
    FTs = []
    STs = []
    for i in range(result.shape[0]):
        FTs.append(np.sum(result[i,:C[i]])/C[i])
        STs.append(np.sum(result[i,:C[i]*2-1])/C[i])
    FT = np.mean(FTs)
    ST = np.mean(STs) 
    # 20-measure
    temp123 = result.sum()
    s = result[:, : 20].sum()
    p = s/(result.shape[0]*20)
    rr = s/temp123
    F_measure = 2.0/(1/p +1/rr)
    
    # DCG
    DCG_child = []
    for i in range(result.shape[0]):
        DCG_k = result[i,1]
        DCG_data = 1
        if C[i] > 2 :
            for k in range(2, C[i]):
                DCG_k += result[i, k]/np.log2(k)
                DCG_data += 1/np.log2(k)
        DCG_child.append(DCG_k/DCG_data)
    DCG = np.sum(DCG_child)/(result.shape[0])

    # ANMRR
    T_max = np.max(C[i])
    NMRR = []
    for i in range(result.shape[0]):
        S_k = np.min([4*C[i],2*T_max])
        r = np.zeros(C[i])
        for k in range(C[i]):
            if result[i,k]==1:
                r[k] = k+1
            else : 
                r[k] = S_k + 1
        NMRR.append((np.sum(r)/C[i] - (C[i]/2-0.5))/(S_k - C[i]/2+0.5))

    ANMRR = np.sum(NMRR)/(result.shape[0])
    evaludation = np.array([NN, FT, ST, F_measure, DCG, ANMRR, mAP])

    #fo =open('pr/%s_%s_%.3f_%.3f.txt' %(name, 'cos' if dist_p else 'elur',mAP,mAP5),'w') 
    #np.savetxt(fo, np.array([r, p]), fmt='%.5f')
    #fo.close() 
    return evaludation


def get_pr(dist_p=0):
    
    fts_qs,las_qs,fts_c,las_c = get_data(p_q, p_c)
    results = []
    for i in tqdm(range(len(np.unique(las_qs)))):
        fts_q = fts_qs[las_qs==i]
        las_q = np.array([i]).repeat(len(fts_q))
        data = [fts_q,las_q,fts_c,las_c]
        evaludation = get_pr_cls(dist_p, data)
        results.append(evaludation)
    #for i in mAP: print('%.3f',end=',') 
    #print('')
    print('NN, FT, ST, F_measure, DCG, ANMRR, mAP')
    print(np.mean(np.stack(results),axis=0)) 

if __name__ == '__main__':

    get_pr(1)
    #pr,mAP,mAP5 = get_pr(0)
    #print('elur map:',mAP,',',mAP5) 
    
    #pr,mAP,mAP5 = get_pr(1)
    #print('cos map:',mAP,',',mAP5) 
    
    
    # print('pr:',pr.shape)
    # plt.plot(pr[0,:],pr[1,:])
    # plt.title('map:%f'%map)
    # plt.show()
