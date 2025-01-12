# GSETRetro
## 0. Requirements
Create a virtual environment and install the dependencies.<br>
Install pytorch with the cuda version that fits your device.<br>
```
conda create -n yourenv python=3.7
conda activate yourenv
conda install rdkit=2020.09.1.0 -c rdkit
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torchtext==0.6.0 torch_geometric==2.3.1 configargparse tensorboardX==2.6.2.2 textdistance==4.2.2 rxnmapper==0.3.0
```
## 1. Data and Checkpoints
The data and checkpoints used to reproduce the results of the paper can be accessed via the link: 
https://drive.google.com/drive/folders/1tHuqdjdu2kxQChS2x-z1n3Hnhf-0rDBt 
- For the Biochem dataset </br>
The raw data is originally sourced from: https://github.com/zengtsysu/BioNavi </br>
Further processed raw data used for this paper is obtained from: https://github.com/SeulLee05/READRetro </br>
- For the USPTO-50K dataset </br>
The raw data is obtained from: https://github.com/Hanjun-Dai/GLN </br>
The 20x augmented data is obtained from: https://github.com/otori-bird/retrosynthesis </br>
The directory structure is shown below, and the relevant files downloaded from the above link should be placed accordingly:
```
GSETRetro
├───GSETransformer
│   ├───data
│   │   ├───biochem_npl_20xaug
│   │   │  └───src-train.txt/tgt-train.txt/src-val.txt/tgt-val.txt …
│   │   │
│   │   └───uspto_50k_20xaug
│   │       └───src-train.txt/tgt-train.txt/src-val.txt/tgt-val.txt …
│   ├───experiments
│   │    ├───biochem_npl_20xaug
│   │    │   └───model_step_xx.pt
│   │    │
│   │    └───uspto_50k_20xaug   
│   │        └───model_step_xx.pt
│   └─……
├───data       
├───retro_star
├───utils      
└───……
```

## 2. Single-step Model Traning and Evaluation
cd /GSETransformer
### (2.1)To preprocess the data used for traning:
```
python preprocess.py -train_src data/biochem_npl_20xaug/src-train.txt -train_tgt data/biochem_npl_20xaug/tgt-train.txt \
                     -valid_src data/biochem_npl_20xaug/src-val.txt  -valid_tgt data/biochem_npl_20xaug/tgt-val.txt  \
                     -save_data data/biochem_npl_20xaug/biochem_npl_20xaug  \
                     -src_seq_length 1000 -tgt_seq_length 1000 -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```
It will generate one `.pt` file and one `.pkl` file each for the train and valid data. </br>
### (2.2)To train the model:
```
CUDA_VISIBLE_DEVICES=${gpu_id}   \
python  train.py -data  data/biochem_npl_20xaug/biochem_npl_20xaug \
                 -save_model experiments/biochem_npl_20xaug/model \
                 -seed 2024 -gpu_ranks 0 \
                 -save_checkpoint_steps 10000  \
                 -train_steps 400000 -valid_steps 1000 -report_every 1000 \
                 -param_init 0 -param_init_glorot \
                 -batch_size 4096 -batch_type tokens -normalization tokens \
                 -dropout 0.3 -max_grad_norm 0 -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 \
                 -decay_method noam -warmup_steps 8000  -learning_rate 2 -label_smoothing 0.0 \
                 -enc_layers 6 -dec_layers 6 -rnn_size 512 -word_vec_size 512 \
                 -encoder_type transformer -decoder_type transformer \
                 -share_embeddings -position_encoding -max_generator_batches 32 \
                 -global_attention general -global_attention_function softmax \
                 -self_attn_type scaled-dot -max_relative_positions 4 \
                 -heads 8 -transformer_ff 2048  -early_stopping 100 -keep_checkpoint 10 \
                 -tensorboard -tensorboard_log_dir runs/biochem_npl_20xaug 2>&1 | tee runs/biochem_npl_20xaug.log
```
### (2.3) To generate prediction and score the output results:
```
CUDA_VISIBLE_DEVICES=${gpu_id}   \
python translate_with_src_aug.py -model experiments/biochem_npl_20xaug/model_best_acc_step_355000.pt   \
                    -src data/biochem_npl_20xaug/src-test.txt -tgt data/biochem_npl_20xaug/tgt-test.txt \
                    -output data/biochem_npl_20xaug/pred.txt -replace_unk  -gpu 0  -beam_size 10 -n_best 10
```
Noted that the hyper-parameter for USPTO-50k: `-beam_size 10 -n_best 50` </br>
if opt.tgt above is given, script will do score the output results automatically.

## 3.Multi-step Planning and Evaluation
### (3.1) To Plan retrosynthetic routes
Run the following command to plan paths of multiple products using multiprocessing:
```
CUDA_VISIBLE_DEVICES=${gpu_id} python run_mp.py
```
### (3.2) To evaluate the result of the previous step
Run the following command to evaluate the planned paths of the test molecules:
```
python eval.py ${save_file}
```
