# Unsupervised Cross-media Graph Convolutional Network for 2D Image based 3D Model Retrieval

Implementation of the paper:  Unsupervised Cross-media Graph Convolutional Network for 2D Image based 3D Model Retrieval.

## 1. Abstract
With the rapid development of 3D construction technology, 3D models have been applied in many applications. In particular, virtual and augmented reality has created a considerable demand for rapid access to large sets of 3D models in recent years. An effective method for addressing the demand is to search 3D models based on 2D images because 2D images can be easily captured by smartphones or other lightweight vision sensors. In this paper, we propose a novel unsupervised cross-media graph convolutional network (UCM-GCN) for 3D model retrieval based on 2D images. Here, we render views from 3D models to construct a graph model based on 3D model structural information. Then, we utilize the 2D image's visual information to bridge the gap between cross-modality data. The operation successfully makes the cross-modality data appear in the same graph. Then, the proposed model, the UCM-GCN model, is utilized to generate node embeddings. Here, we introduce correlation loss to mitigate the distribution discrepancy across different modalities, which can fully consider the structural and visual similarities between the 2D image and 3D model to make the final different modalities embed into the same feature space. To demonstrate the performance of our approach, we conducted a series of experiments on the MI3DOR dataset, which is utilized in Shrec19. We also compared it with other similar methods in the 3D-FUTURE dataset. The experimental results demonstrate the superiority of our proposed method over state-of-the-art methods.

## 2. Contributions
- We design a cross-modality graph model for 2D-image-based 3D model retrieval, which fully utilizes the visual information of the image and structural information of the 3D model to bridge the 2D image and 3D model;
- We propose a novel unsupervised GCN model to align the feature distribution from different modalities. More specifically, we apply the GCN to update the descriptors according to the cross-media graph, which is designed to narrow the gap between them;
- We conduct corresponding experiments on the MI3DOR dataset to evaluate the performance of our approach. The experimental results demonstrate the superiority of the proposed method.


![](https://github.com/tjuliangqi/MI3DOR_graph/blob/main/pic/framework.png)

## 3. Approach
In this section, we detail the framework of our approach. Fig.2 shows the pipeline of the UCM-GCN. the proposed method mainly includes three consecutive steps.

- 1) Data processing: we extract the rendered views for each 3D model from different angles. We hope these rendered views can include as many angles as possible. The pre-trained CNN model \cite{efficentnet} is utilized to extract the visual feature vector for the 2D image and the rendered views.
- 2) Graph Construction: this goal of this step is to utilize the visual similarity between the 2D image and rendered view to bridge the gap between image and 3D model. It directly influences the performance of graph embedding. Thus, we consider the visual similarity and structure similarity to build the graph.
- 3) Graph Embedding: we apply the classic GCN model and introduce the correlation loss to generate the embeddings of nodes, which are used to compute the similarity between the 2D image and 3D model for retrieval problem.

## 4. Authors
- Qi Liang
- Qiang Li 
- Weizhi Nie
- An-An Liu
- Yuting Su

## 5. Usage
`mkdir data` 

`mkdir checkpoints`

`Run the train.sh.`

