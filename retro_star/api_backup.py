import torch
import pandas as pd
from retro_star.common import prepare_starting_molecules, \
     prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
# from g2s.translate import prepare_g2s, run_g2s
from GSETransformer.translate_multi_step import prepare_GSET_translator, run_GSET_translate
# from retroformer.translate import prepare_retroformer, run_retroformer
# from megan.translate import prepare_megan, run_megan
# from utils.ensemble import prepare_ensemble, run_ensemble
from retro_star.retriever import Retriever, run_retriever_GSET
# from retro_star.retriever import Retriever, run_retriever, run_retriever_only, \
#     neutralize_atoms, kegg_search, \
from retro_star.retriever import neutralize_atoms, kegg_search
#     pathRetriever, run_path_retriever, run_both_retriever
from rdkit import Chem


class RSPlanner:
    def __init__(self,
                 cuda=True,
                 beam_size=10,
                 iterations=500,
                 expansion_topk=10,
                 use_value_fn=True,
                 starting_molecules='data/building_block.csv',
                 value_model='retro_star/saved_model/best_epoch_final_4.pt',
                 model_type='ensemble',
                 model_path=None,
                 retrieval_db='data/train_canonicalized.txt',
                 path_retrieval_db='data/pathways.pickle',
                 kegg_mol_db = "data/kegg_neutral_iso_smi.csv",
                 route_topk=10,
                 retrieval=True,
                 path_retrieval=True):
        
        setup_logger()
        device = torch.device('cuda' if cuda else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)
        self.path_retrieve_token = None
        self.kegg_mol_db = None
        self.path_retrieval_db = None

        self.supplemental_model = False
        print(f'self.supplemental_model {self.supplemental_model }')
        gset_model_path = '/home/zhangmeng/aMy-ONMT003/experiments/biochem_npl_10xaug_RSmiles_sdd_token/model_step_355000.pt'
        gset_model_path2 = '/home/zhangmeng/aMy-ONMT003/experiments/biochem_npl_10xaug_RSmiles_sdd_token/model_step_355000.pt'

        if model_path is None:
            retroformer_path = 'retroformer/saved_models/biochem.pt'
            g2s_path = 'g2s/saved_models/biochem.pt'
        
        if model_type == 'retroformer':
            if model_path is not None:
                retroformer_path = model_path


        elif model_type == 'GSETransformer':
            # if model_path is None:
            #     gset_model_path = ['/home/zhangmeng/aMy-ONMT003/experiments/biochem_npl_10xaug_RSmiles_6x512/model_step_avg_375000.pt']

            translator, opt = prepare_GSET_translator(gset_model_path,  beam_size, expansion_topk, device)
            expansion_handler = lambda x: run_GSET_translate(x, translator, opt)
            #################
            translator2, opt2 = prepare_GSET_translator(gset_model_path2, beam_size, expansion_topk, device)
            expansion_handler2 = lambda x: run_GSET_translate(x, translator2, opt2)


        elif not path_retrieval and retrieval:  # READRetro
            # if model_path is not None:
            #     retroformer_path, g2s_path = model_path.split(',')

            translator, opt = prepare_GSET_translator(gset_model_path, beam_size, expansion_topk, device)
            retriever = Retriever(retrieval_db)
            expansion_handler = lambda x: run_retriever_GSET(x, retriever, translator, opt)
            ###
            translator2, opt2 = prepare_GSET_translator(gset_model_path2, beam_size, expansion_topk, device)
            expansion_handler2 = lambda x: run_retriever_GSET(x, retriever, translator2, opt2)




        self.top_k = route_topk

        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=2048,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model.load_state_dict(torch.load(value_model, map_location=device))
            model.eval()

            def value_fn(mol, retrieved):
                if retrieved: return 0.
                # import pdb; pdb.set_trace()
                fp = smiles_to_fp(mol, fp_dim=2048).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x, r: 0.

        self.plan_handle = prepare_molstar_planner(
            expansion_handler=expansion_handler,
            value_fn=value_fn,
            starting_mols=starting_mols,
            iterations=iterations
        )
        self.plan_handle2 = prepare_molstar_planner(
            expansion_handler=expansion_handler2,
            value_fn=value_fn,
            starting_mols=starting_mols,
            iterations=iterations
        )

    def __keggpath_find(self,routes,token,mol_db,path_db,top_k):
        modi_list = []
        for route in routes[:top_k]:
            r = route.split(">")
            token_position = [i for i,j in enumerate(r) if token in j]
            for pos in token_position:
                cid, _ = kegg_search(neutralize_atoms(r[pos-2].split("|")[-1]),mol_db)
                target_maps = path_db["Map"][path_db['Pathways'].apply(lambda x: any(cid in sublist for sublist in x))].to_list()
                map = target_maps[0]  # check a representation method
                r[pos] = r[pos].replace(token,f'{token}=kegg.jp/pathway/{map}+{cid}')
                if target_maps == []:  # not the case
                    modi_list.append(route)

            modi_route = '>'.join(r)
            modi_list.append(modi_route)
        return modi_list
    def __single_model_plan(self, plan_handle, target_mol_in):
        try:
            target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol_in))
            succ, msg = plan_handle(target_mol)    # the result of model

            if succ:
                routes_list = []
                for route in msg:
                    routes_list.append(route.serialize())
                return succ, routes_list[:self.top_k]#[:self.top_k]  # ,modi_routes_list

            elif target_mol != Chem.MolToSmiles(Chem.MolFromSmiles(target_mol), isomericSmiles=False):
                no_stereo_target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol), isomericSmiles=False)
                succ, msg = plan_handle(no_stereo_target_mol)    # the result of model
                if succ:
                    routes_list = []
                    for route in msg:
                        routes_list.append(route.serialize())
                    modified_routes_list = []
                    for route_seq in routes_list:
                        modified_route = route_seq.replace(no_stereo_target_mol+'>', target_mol+'>')
                        modified_routes_list.append(modified_route)
                    return succ, modified_routes_list[:self.top_k]#[:self.top_k]#modi_routes_list

            else:
                return succ, None
        except:
            print(f'invalid smiles:{target_mol_in}')
            return False, None

    def plan(self, target_mol_in):
        # target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol))
        # succ, msg = self.plan_handle(target_mol)  # the result of model
        #
        # if succ:
        #     routes_list = []
        #     for route in msg:
        #         routes_list.append(route.serialize())
        #     # modi_routes_list = self.__keggpath_find(routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
        #     return routes_list  # ,modi_routes_list
        # else:
        #     return None
        succ, routes_list = self.__single_model_plan(self.plan_handle, target_mol_in)
        if not succ and self.supplemental_model:
            succ, routes_list = self.__single_model_plan(self.plan_handle2, target_mol_in)
        return routes_list
        # try:
        #     target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol_in))
        #     succ, msg = self.plan_handle(target_mol)    # the result of model
        #
        #     if succ:
        #         routes_list = []
        #         for route in msg:
        #             routes_list.append(route.serialize())
        #         # modi_routes_list = self.__keggpath_find(routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
        #         return routes_list[:self.top_k]#[:self.top_k]  # ,modi_routes_list
        #         # print('11111111111111111111111111111111111111111111111111111111111111111111111111')
        #         # print(modi_routes_list)
        #         # return modi_routes_list
        #
        #     elif target_mol != Chem.MolToSmiles(Chem.MolFromSmiles(target_mol), isomericSmiles=False):
        #         no_stereo_target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol), isomericSmiles=False)
        #         succ, msg = self.plan_handle(no_stereo_target_mol)    # the result of model
        #         if succ:
        #             routes_list = []
        #             for route in msg:
        #                 routes_list.append(route.serialize())
        #             modified_routes_list = []
        #             for route_seq in routes_list:
        #                 modified_route = route_seq.replace(no_stereo_target_mol+'>', target_mol+'>')
        #                 modified_routes_list.append(modified_route)
        #             #modi_routes_list = self.__keggpath_find(routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
        #             # modi_routes_list = self.__keggpath_find(modified_routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
        #             # print(routes_list)
        #             # print(modified_routes_list)
        #             # return modi_routes_list
        #             return modified_routes_list[:self.top_k]#[:self.top_k]#modi_routes_list
        #
        #     else:
        #         if self.supplemental_model:
        #             try:
        #                 target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol_in))
        #                 succ, msg = self.plan_handle2(target_mol)  # the result of model
        #                 if succ:
        #                     routes_list = []
        #                     for route in msg:
        #                         routes_list.append(route.serialize())
        #                     # modi_routes_list = self.__keggpath_find(routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
        #                     return routes_list[:self.top_k]
        #                 else:
        #                     return None
        #             except:
        #                 return None
        #         else:
        #             return None
        # except:
        #     print(f'invalid smiles:{target_mol_in}')
        #     return None