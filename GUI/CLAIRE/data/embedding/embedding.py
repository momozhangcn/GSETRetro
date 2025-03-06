
"""
生成 rxnfp 和 drfp 的embedding
"""

### rxnfp embedding
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
import pickle

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

example_rxn = ["C/C(C=O)=C\CC/C(C)=C/C=O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.[H+].O>>COC(=O)CCCCCCCC=O", "CC(C=O)c1ccccc1.O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC(C(=O)O)c1ccccc1"]

fp = rxnfp_generator.convert_batch(example_rxn)
print(len(fp))

esm_emb_dict = {}
for ind, rxn in enumerate(example_rxn):
    esm_emb_dict[rxn] = fp[ind]
path = './data/inputs/rxn_ec_emb.pkl'
with open(path, 'wb') as file:
    pickle.dump(esm_emb_dict, file)
print(len(esm_emb_dict))


### drfp embedding

## 安装 drfp 环境
# pip install drfp

# 命令行运行命令：drfp my_rxn_smiles.txt my_rxn_fps.pkl -d 256
# 其中，将my_rxn_smiles.txt替换为自己的rxn序列文件地址，my_rxn_fps.pkl 替换为保存的embedding的地址。
# -d 为生成的embedding维度


