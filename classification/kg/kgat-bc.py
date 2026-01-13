import dgl
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from dgl.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_and_build_graph(csv_path):
    """从CSV文件构建知识图谱"""
    df = pd.read_csv(CSV_PATH, header=None, names=["head", "rel", "tail"])

    # 创建实体和关系映射
    entities = sorted(list(set(df["head"].unique().tolist() + df["tail"].unique().tolist())))
    relations = sorted(df["rel"].unique().tolist())

    entity2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    # 构建DGL图
    g = dgl.DGLGraph()
    g.add_nodes(len(entities))

    # 添加边
    src = [entity2id[h] for h in df["head"]]
    dst = [entity2id[t] for t in df["tail"]]
    rel_ids = [rel2id[r] for r in df["rel"]]

    g.add_edges(src, dst)
    g.edata["rel_type"] = torch.tensor(rel_ids)  # 边关系类型
    g = dgl.add_self_loop(g)

    return g, entities, relations, entity2id


# ----------------------------
# Step 2: PubMedBERT特征提取
# ----------------------------
class PubMedBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("E:/Pycharm/myProject/kg-emb/pubmedbert/")
        self.model = AutoModel.from_pretrained("E:/Pycharm/myProject/kg-emb/pubmedbert")

    def get_entity_embeddings(self, entities, batch_size=8):
        """批量生成实体文本嵌入"""
        self.model.eval()
        embeddings = []

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            embeddings.append(batch_emb)

        return torch.cat(embeddings, dim=0)


# ----------------------------
# Step 3: GAT模型定义
# ----------------------------
class PubMedGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, features):
        h = self.gat1(g, features)  # [N, num_heads, hidden_dim]
        h = h.view(h.size(0), -1)  # [N, num_heads*hidden_dim]
        h = F.elu(h)
        h = self.gat2(g, h)  # [N, out_dim]
        return h


# ----------------------------
# Step 4: 执行流程
# ----------------------------
if __name__ == "__main__":
    # 参数设置
    CSV_PATH = "E:/Pycharm/myProject/kg-emb/data/kg4bert_p.csv" # 替换为你的CSV路径
    EMB_DIM = 512  # 最终嵌入维度

    # 1. 构建知识图谱
    g, entities, _ ,e2id= load_and_build_graph(CSV_PATH)

    # 2. 使用PubMedBERT生成初始特征
    bert_encoder = PubMedBERTEncoder()
    bert_embeddings = bert_encoder.get_entity_embeddings(entities)
    print(f"PubMedBERT特征维度: {bert_embeddings.shape}")  # [N, 768]

    # 3. 定义GAT模型（将768维BERT特征压缩到目标维度）
    gat_model = PubMedGAT(
        in_dim=768,
        hidden_dim=512,
        out_dim=EMB_DIM,
        num_heads=4
    )

    # 4. 生成图嵌入
    with torch.no_grad():
        kg_embeddings = gat_model(g, bert_embeddings)

    # 5. 保存结果
    entity_emb_map = {
        ent: emb.numpy()
        for ent, emb in zip(entities, kg_embeddings)
    }
    np.save("entity_embeddings_p.npy", entity_emb_map)
    print("嵌入已保存至 entity_embeddings_p.npy")

    #生成患者词向量
    target_entity_names = ["S0000004","S0000015","S0000110","S0000145","S0000579","S0000821","S0000847","S0000909","S0002212","S0002267","S0003674","S0005019","S0005424","S0005513","S0005574","S0006017","S0006214","S0006838","S0007051","S0007310","S0007349","S0009135","S0010111","S0010539","S0010548","S0011433","S0011644","S0012014","S0012077","S0012156","S0000091","S0000960","S0001016","S0001025","S0001435","S0001794","S0002246","S0002300","S0002544","S0002581","S0002680","S0002739","S0003062","S0003156","S0003342","S0003352","S0003427","S0003596","S0003749","S0003877","S0003913","S0003918","S0004002","S0004003","S0004088","S0004095","S0004103","S0001511","S0001780","S0002597","S0003423","S0003573","S0003798","S0003910","S0004089","S0004091","S0004176","S0004540","S0004585","S0004760","S0004889","S0005012","S0005099","S0000151","S0000152","S0000170","S0000247","S0000445","S0000660","S0000784","S0000828","S0001260","S0002696","S0002780","S0002833","S0002936","S0002997","S0003181","S0003481","S0003707","S0015838","S0016183","S0016583","S0016596","S0016647","S0016732","S0016733","S0016735","S0016744","S0016752","S0016815","S0004027","S0004265","S0004275","S0004359","S0004362","S0004794","S0005648","S0005926","S0005984","S0006123","S0006336","S0006439","S0006562","S0006655","S0005812","S0007527","S0007529","S0007940","S0006357","S0006571","S0006652","S0006865","S0006870","S0004267","S0004852","S0004970","S0002922","S0005723","S0005772","S0005806","S0005839","S0005842","S0004538","S0004815","S0005271","S0005350","S0005159","S0006012","S0006026","S0006133","S0006251","S0004266","S0007251","S0004844","S0004899","S0005061","S0005261","S0004684","S0004255","S0004200","S0004104","S0016018","S0015989","S0007040","S0006216","S0006550","S0004363","S0004268","S0004781","S0004782","S0000608","S0004445","S0004110","S0004603","S0001506","S0003013","S0003093","S0003094","S0003178","S0003287","S0003371","S0003634",]
    target_indices = [e2id[name] for name in target_entity_names if name in e2id]
    print(f"目标实体: {target_entity_names}")
    print(f"对应的索引: {target_indices}")

    target_embeddings = kg_embeddings[target_indices]

    print(f"提取到的嵌入向量形状: {target_embeddings.shape}")
    with open('target_embeddings.pkl', 'wb') as f:
        pickle.dump(target_embeddings, f)

    print("患者词向量计算与存储完成。")
    #torch.save(target_embeddings, 'target_entity_embeddings.pt')

'''
# 1. 读取患者特征数据
# 假设CSV文件无表头，列分别为患者ID和特征
df = pd.read_csv('E:/Pycharm/myProject/kg-emb/data/patient_features.csv', header=0, names=['patient_id', 'feature'])

# 按患者ID分组，收集特征列表
patient_groups = df.groupby('patient_id')['feature'].apply(list)
#print(patient_groups.items)


# 2. 加载知识图谱词向量
kg_emb = np.load('entity_embeddings.npy',allow_pickle=True)
emb_dict = np.load("entity_embeddings.npy", allow_pickle=True).item()


# 3. 计算每位患者的平均词向量

patient_embeddings = {}
for patient_id, features in patient_groups.items():
    print(patient_id,features)
    vectors = []
    for feature in features:
        #print(feature)
        patient_emb = emb_dict[feature]
        #print(patient_emb)
        vectors.append(patient_emb)

    if not vectors:
        print(f"警告：患者 {patient_id} 无有效特征，使用零向量。")
        avg_vec = np.zeros(kg_emb.shape[1])
    else:
        avg_vec = np.mean(vectors, axis=0)

    patient_embeddings[patient_id] = avg_vec
    print(patient_embeddings)

# 4. 存储患者词向量（使用pickle保存字典）
with open('patient_embeddings.pkl', 'wb') as f:
    pickle.dump(patient_embeddings, f)

print("患者词向量计算与存储完成。")

'''


