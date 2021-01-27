# coding utf-8
'''
代码参考了作者Laurens van der Maaten的开放出的t-sne代码, 并没有用类进行实现,主要是优化了计算的实现
'''
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^w + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)+1e-4
    return dist


def cal_perplexity(dist, idx=0, beta=1.0):
    '''计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    这里的perp仅计算了熵，方便计算
    '''

    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)

    if sum_prob==0:
        sum_prob=0.01
        print('----',sum_prob==0)
    perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
    prob /= sum_prob
    return perp, prob


def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''二分搜索寻找beta,并计算pairwise的prob
    '''

    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        #print('+',i,end=',')
        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return pair_prob


def pca(x, no_dims=50):
    ''' PCA算法
    使用PCA先进行预降维
    '''
    print("Preprocessing the data using PCA...")
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1))
    l, M = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, M[:, 0:no_dims])
    return y


def tsne(x, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # 初始化参数和变量
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # 对称化
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration
    P = P * 4
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum(P / 4 * np.log(P / 4 / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


if __name__ == "__main__":
    # Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.
    # =============== hyper parameter
    ag_epoch = sys.argv[1]
    cls_num = 10 
    iter_num = 1000
    eps = False
    dark_color = np.array(['#00ff00', '#0000ff', '#ff0000', '#ffff00', '#00ffff', '#ff00ff', '#191717'])
    light_color = np.array(['#abf5ab', '#9090fb', '#ff9b9b', '#f0f094', '#87f5f5', '#ff9aff', '#aaa9a9'])

   
    
    
    #====================read data
    #X = loadmat(path[4])
    path = ['feature/%s/sketch_feat.npy' % (ag_epoch),\
            'feature/%s/model_feat.npy' % (ag_epoch)]
    #path = ['sketch_feat.npy' ,\
    #        'model_feat.npy' ]
    X_s = np.load(path[0], allow_pickle=True).item() 
    X_m = np.load(path[1], allow_pickle=True).item() 
    

    # ===================choice data
    choice = np.random.choice(range(2700),1000,replace=False)
    
    bool_s = X_s['las']<10 #[choice] 
    bool_m = X_m['las']<10 #[choice] 
    
    labels_s = X_s['las'][bool_s] #[choice] 
    labels_m = X_m['las'][bool_m]
    len_s = len(labels_s)
    labels = np.concatenate((labels_s, labels_m), axis=0)

    X_s = X_s['fts'][bool_s] #[choice]
    X_m = X_m['fts'][bool_m] #[choice]
    X = np.concatenate((X_s, X_m), axis=0) 
    #color = np.array(['blue','orange','green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 100, 20.0, max_iter=iter_num)
     
    
    from matplotlib import pyplot as plt

    plt.scatter(Y[:len_s, 0], Y[:len_s, 1], 10, labels_s,cmap=plt.get_cmap('spring'))  #dark_color[labels_s])
    x=[Y[:len_s,0][(labels_s==i).astype(bool)].mean()  for i in range(cls_num)]
    y=[Y[:len_s,1][(labels_s==i).astype(bool)].mean()  for i in range(cls_num)]
    
    for i in range(cls_num):
        plt.text(x[i],y[i],'%d'%i)
    #plt.show()
    if eps:
        plt.savefig('tsne_sketch.eps')
    else :
        plt.savefig('tsne_sketch.jpg')
    print('save tsne model ...')
    plt.close()
    
    plt.scatter(Y[len_s:, 0], Y[len_s:, 1], 10, labels_m, cmap=plt.get_cmap('winter')) #light_color[labels_m])
    x=[Y[len_s:, 0][(labels_m==i).astype(bool)].mean()  for i in range(cls_num)]
    y=[Y[len_s:, 1][(labels_m==i).astype(bool)].mean()  for i in range(cls_num)]
    #print()
    for i in range(cls_num):
        plt.text(x[i],y[i],'%d'%i)
    #plt.show()
    if eps:
        plt.savefig('tsne_model.eps')
    else :
        plt.savefig('tsne_model.jpg')
    print('save tsne image ...')
    plt.close()

    
    plt.scatter(Y[:len_s, 0], Y[:len_s, 1], 10, labels_s,cmap=plt.get_cmap('spring'))
    plt.scatter(Y[len_s:, 0], Y[len_s:, 1], 10, labels_m,cmap=plt.get_cmap('winter'))
    
    x=[Y[:len_s, 0][(labels_s==i).astype(bool)].mean()  for i in range(cls_num)]
    y=[Y[:len_s, 1][(labels_s==i).astype(bool)].mean()  for i in range(cls_num)]
    #print()
    for i in range(cls_num):
        plt.text(x[i],y[i],'%d'%i)
    
    x=[Y[len_s:, 0][(labels_m==i).astype(bool)].mean()  for i in range(cls_num)]
    y=[Y[len_s:, 1][(labels_m==i).astype(bool)].mean()  for i in range(cls_num)]
    #print()
    for i in range(cls_num):
        plt.text(x[i],y[i],'%d'%i)
    #plt.show()
    if eps:
        plt.savefig('tsne_compare.eps')
    else :
        plt.savefig('tsne_compare.jpg')
    print('save tsne double...')
    plt.close()
