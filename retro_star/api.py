import torch
import os
from retro_star.common import prepare_starting_molecules, \
     prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from rdkit import Chem
import random

class RSPlanner:
    root_dir_path = os.path.dirname(os.path.dirname(__file__))
    def __init__(self,
                 cuda=True,
                 beam_size=10,
                 iterations=500,
                 expansion_topk=10,
                 use_value_fn=True,
                 starting_molecules=f'{root_dir_path}/data/building_block.csv',
                 value_model=f'{os.path.dirname(__file__)}/saved_model/best_epoch_final_4.pt',
                 model_type='ensemble',
                 model_path=None,
                 retrieval_db=f'{root_dir_path}/data/train_canonicalized.txt',
                 path_retrieval_db='data/pathways.pickle',
                 kegg_mol_db="data/kegg_neutral_iso_smi.csv",
                 route_topk=10,
                 retrieval=True,
                 path_retrieval=True,
                 seed=3435):

        setup_logger()
        if cuda and torch.cuda.is_available():
            device = torch.device('cuda')
            device_bool = True
        else:
            device = torch.device('cpu')
            device_bool = False
        random.seed(seed)
        torch.manual_seed(seed)

        starting_mols = prepare_starting_molecules(starting_molecules)

        self.path_retrieve_token = None
        self.kegg_mol_db = None
        self.path_retrieval_db = None

        if model_path is not None:
            gset_model_path = model_path
        else:
            gset_model_path = '/home/zhangmeng/aMy-ONMT010/experiments/final_biochem_npl_20xaug_RSmiles/model_best_acc_step_355000.pt'

        if model_type == 'Chemformer': #prepare_GSET_translator, run_GSET_translate
            from Chemformer.translate_multi_step import prepare_MolBART_translator, run_MolBART_translate
            #MolBART_model_path = '/data/zhangmeng/Chemformer-main/tb_logs/backward_prediction/ori_data_new_vocab_100e/checkpoints/epoch=49-step=34699.ckpt'
            MolBART_model_path = '/data/zhangmeng/Chemformer-main/tb_logs/backward_prediction/version_2/checkpoints/epoch=199-step=137799.ckpt'
            load_model, tokeniser, args = prepare_MolBART_translator(MolBART_model_path, beam_size, expansion_topk, device)
            expansion_handler = lambda x: run_MolBART_translate(x, load_model, tokeniser, args)

        elif model_type == 'AugTransformer':
            from AugTransformer.translate_multi_step import prepare_onmt_translator, run_onmt_translate
            #onmt_model_path = '/home/zhangmeng/OpenNMT-py-0.9.1/experiments/final_biochem_npl_20xaug_RSmiles/model_best_ppl_step_140000.pt'
            onmt_model_path = '/home/zhangmeng/OpenNMT-py-0.9.1/experiments/final_biochem_npl_clean_20xaug_Cano/model_best_ppl_step_155000.pt'
            translator, opt = prepare_onmt_translator(onmt_model_path,  beam_size, expansion_topk, device)
            print(onmt_model_path)
            expansion_handler = lambda x: run_onmt_translate(x, translator, opt)

        elif model_type == 'TTWTransformer':
            from TTWTransformer.translate_multi_step import prepare_TTWTransformer_translator, \
                run_TTWTransformer_translate
            #model_path = '/data/zhangmeng/tied-twoway-transformer-main/onmt-runs/sdd_biochem_npl/model_step_150000.pt'
            model_path = '/data/zhangmeng/tied-twoway-transformer-main/onmt-runs/sdd_biochem_npl_clean/model_step_70000.pt'
            translator_x2y, opt = prepare_TTWTransformer_translator(model_path, beam_size, expansion_topk, device)
            translator_y2x, opt = prepare_TTWTransformer_translator(model_path, beam_size, expansion_topk, device)
            expansion_handler = lambda x: run_TTWTransformer_translate(x, translator_x2y, translator_y2x, opt)

        elif model_type == 'GTA':
            from GTA.translate_multi_step import prepare_GTA_translator, run_GTA_translate
            #model_path = '/data/zhangmeng/GTA-master/experiments/biochem_can_be_mapped_sep_token/models/GTA_model4_bio_avg10w-15w.pt'
            model_path = '/data/zhangmeng/GTA-master/experiments/sdd_biochem_npl/models/model_step_160000.pt'
            translator, opt = prepare_GTA_translator(model_path,  beam_size, expansion_topk, device)
            expansion_handler = lambda x: run_GTA_translate(x, translator, opt)

        elif model_type == 'megan':
            from megan.translate import prepare_megan, run_megan
            model_path = '/data/zhangmeng/GSETRetro-main/megan/model_megan/saved_models/biochem_model_best.pt'
            model_megan, action_vocab, base_action_masks = prepare_megan(cuda,
                                                                         path=model_path)
            expansion_handler = lambda x: run_megan(model_megan, action_vocab, base_action_masks, x)

        elif model_type == 'GSETransformer':
            from GSETransformer.translate_multi_step import prepare_GSET_translator, run_GSET_translate
            translator, opt = prepare_GSET_translator(gset_model_path,  beam_size, expansion_topk, device_bool)
            expansion_handler = lambda x: run_GSET_translate(x, translator, opt)

        #else:
        elif model_type in ['ensemble', 'GSETransformer+Retriver']:  # READRetro
            from GSETransformer.translate_multi_step import prepare_GSET_translator
            from retro_star.retriever import Retriever, run_retriever_GSET
            translator, opt = prepare_GSET_translator(gset_model_path, beam_size, expansion_topk, device)
            retriever = Retriever(retrieval_db)
            expansion_handler = lambda x: run_retriever_GSET(x, retriever, translator, opt)

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

    def _single_plan_handle(self, plan_handle, target_mol_in):
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

            else:
                return succ, None
        except:
            print(f'invalid smiles:{target_mol_in}')
            return False, None

    def plan(self, target_mol_in):
        succ, routes_list = self._single_plan_handle(self.plan_handle, target_mol_in)
        if succ:
            print(routes_list)
        return routes_list