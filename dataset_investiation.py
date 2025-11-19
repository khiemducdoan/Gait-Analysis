import pandas as pd
import numpy as np
from dataloader.dataloaders import *
import argparse
import wandb
import optuna
import joblib
import shutil
import glob
import datetime 
from model.motion_encoder import MotionEncoder
from model.backbone_loader import load_pretrained_backbone, count_parameters, load_pretrained_weights
from train import train_model, validate_model
from const import path, const
from learning.utils import log_cfm_to_wandb
from utility import utils
from utility.utils import set_random_seed, override_dataset
from configs import generate_config_motionbert, generate_config_poseformerv2, generate_config_mixste, generate_config_motionagformer, generate_config_momask, generate_config_motionclip, generate_config_potr 
def main(params):
    print(params)
    backbone_name = params["backbone"]
    train_dataset, eval_dataset = dataset_factory_custom(params, backbone_name, 1)
    train_dataset_fn, _, class_weights = dataloader_factory_custom(params, train_dataset, eval_dataset, eval_batch_size= 1)
def get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, fold, augmented_datasets=False):
    if not params['LODO']:
        train_dataset, eval_dataset = dataset_factory(params, backbone_name, fold)
    else:
        if params['AID'] and not augmented_datasets:
            # This should be LOSO as it is the in domain dataset
            assert params['num_folds'] == const.NUM_OF_PATIENTS_PER_DATASET[params['dataset']], "AID is only supported for LOSO"
            train_dataset, eval_dataset = dataset_factory(params, backbone_name, fold)
        else:
            if params['AID'] and augmented_datasets:
                nn_params = params.copy()
                nn_params['num_folds'] = 6
                other_datasets = [d for d in const.SUPPORTED_DATASETS if d != params['dataset']]
                other_datasets = [dataset_factory(override_dataset(nn_params, d), backbone_name, fold) for d in other_datasets]
                train_dataset = torch.utils.data.ConcatDataset([x[0] for x in other_datasets])
                eval_dataset  = torch.utils.data.ConcatDataset([x[1] for x in other_datasets])
            else:
                other_datasets = [d for d in const.SUPPORTED_DATASETS if d != params['dataset']]
                other_datasets = [dataset_factory(override_dataset(params, d), backbone_name, fold) for d in other_datasets]
                train_dataset = torch.utils.data.ConcatDataset([x[0] for x in other_datasets])
                eval_dataset  = torch.utils.data.ConcatDataset([x[1] for x in other_datasets])
                
    return train_dataset, eval_dataset
class SaveStudyCallback:
    def __init__(self, save_frequency, file_path):
        self.save_frequency = save_frequency
        self.file_path = file_path
        self.trial_counter = 0
        if not os.path.exists(self.file_path.split('study_mid.pkl')[0]):
            os.mkdir(self.file_path.split('study_mid.pkl')[0])

    def __call__(self, study, trial):
        self.trial_counter += 1
        if self.trial_counter % self.save_frequency == 0:
            joblib.dump(study, self.file_path)
        print('⭐️'*80)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone', type=str, default='motionbert', help='model name (motionbert, potr, mixste, poseformerv2, motionagformer, momask, motionclip)')
    parser.add_argument('--config', type=str, default=None, help='if left as None all configs will be processed, if it is set to a specific config file (with extension) only it will be processed.')
    parser.add_argument('--train_mode', type=str, default='classifier_only', help='train mode( end2end, classifier_only )')
    parser.add_argument('--num_folds', type=int, default=6, help='Which cv variant to use. If -1, LOSO is performed')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int, help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--ntrials', default=30, type=int, help='number of hyper-param tuning trials')
    parser.add_argument('--this_run_num', type=str, help='Prefix for folder in which results specific to this run should be outputted.')
    parser.add_argument('--readstudyfrom', type=int)
    parser.add_argument('--hypertune', default=1, type=int, help='perform hyper parameter tuning [0 or 1]')
    parser.add_argument('--just_gen_dataset', default=0, type=int, help='When set to 1 only the generation of .pkl dataset files will be initiated.')
    parser.add_argument('--cross_dataset_test', type=int, help='perform testing on --dataset flag dataset (0), or perform testing on all other supported datasets (1) [0 or 1]. Only matters when --hypertune=0')
    parser.add_argument('--pretrained', default=0, type=int, help='If 1 training will be skipped during testing and already existing checkpoints will be used.')
    parser.add_argument('--overwrite_results', default=0, type=int, help='If 1 computed results during model evaluation will be computed again and overwrite existing results')
    parser.add_argument('--force_LODO', default=0, type=int, help='If 1 all config files will have LODO overriden to True and the model_prefix will have the _LODO suffix added if it is not already present')
    parser.add_argument('--AID', default=0, type=int, help='If 1 the model will be trained with Augmented-In-Domain set-up')
    
    parser.add_argument('--combine_views_preds', default=0, type=int, help='If 1 the predictions of all views will be combined **(Make sure you pass --views_configs as well)**')
    parser.add_argument('--views_path', default=None, type=str, nargs='+', help='List of view file paths, First give BACK path and then SIDE path')
    parser.add_argument('--exp_name_rigid', default=None, type=str, help='if you want to override the experiment name in the config file')
    parser.add_argument('--prefer_right', default=0, type=int, help='Prefer right side view for prediction aggregation [0 or 1]')
    
    parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    parser.add_argument('--metadata', default='', type=str, help="add metadata prob to the training 'gender,age,bmi,height,weight'")
    parser.add_argument('--tuned_model_config', type=str, default=None, help='Path to a JSON file containing best hyperparameters if no study.pkl is found.')


    args = parser.parse_args()
    
    params = vars(args)
    # print("\n" + "="*40)
    # print("🔹 Parameters Configuration 🔹")
    # print("="*40)
    # print(json.dumps(param, indent=4))
    # print("="*40 + "\n")
    
    # param['metadata'] = param['metadata'].split(',') if param['metadata'] else []
    
    # torch.backends.cudnn.benchmark = False
    
    # backbone_name = param['backbone']
    # conf_path = const.BACKBONE_CONFIGS.get(backbone_name)
    # if conf_path is None:
    #     raise NotImplementedError(f"Backbone '{backbone_name}' is not supported")
    
    # backbone_config_generators = {
    #     'potr': generate_config_potr.generate_config,
    #     'motionbert': generate_config_motionbert.generate_config,
    #     'poseformerv2': generate_config_poseformerv2.generate_config,
    #     'mixste': generate_config_mixste.generate_config,
    #     'motionagformer': generate_config_motionagformer.generate_config,
    #     'momask': generate_config_momask.generate_config,
    #     'motionclip': generate_config_motionclip.generate_config
    # }
    
    # config_list = sorted(os.listdir(conf_path))
    # if param['combine_views_preds']:
    #     assert param['views_path'] is not None, "If --combine_views_preds is set to 1, --views_configs must be provided with the config files for each view."
    #     config_list = []
    #     config_list.append(glob.glob(os.path.join(path.OUT_PATH, param['views_path'][0], 'config', 'originalfile_*.json'))[0])
    #     config_list.append(glob.glob(os.path.join(path.OUT_PATH, param['views_path'][1], 'config', 'originalfile_*.json'))[0]) 
    #     view_out_pathes = []

    # for fi in config_list:
    #     if param['config'] is not None and param['config'] != fi and not param['combine_views_preds']: continue # If config was specified only run that config
        
    #     generate_config_func = backbone_config_generators.get(backbone_name)
    #     if generate_config_func is None:
    #         raise NotImplementedError(f"Backbone '{backbone_name}' does not exist!")

    #     params, new_params = generate_config_func(param, fi)
    #     if param['exp_name_rigid'] is not None:
    #         params['experiment_name'] = param['exp_name_rigid']
        
    #     # == Determine number of folds ==
    #     if params['num_folds'] == -1: 
    #         if params['LODO'] and not params['AID']: raise NotImplementedError('num_folds is -1 (which means LOSO). This is not implemented when performing LODO but not AID.')
    #         params['num_folds'] = const.NUM_OF_PATIENTS_PER_DATASET[params['dataset']]
        
    #     # == Determine number of classes ==
    #     if params['dataset'] in const.SUPPORTED_DATASETS:
    #         if not params['LODO']:
    #             params['num_classes'] = const.NUM_CLASSES_PER_DATASET[params['dataset']]
    #         else:
    #             params['num_classes'] = int(np.max([n for d,n in const.NUM_CLASSES_PER_DATASET.items() if d != params['dataset']]))
    #     else:
    #         print(f"Dataset '{params['dataset']}' not found in SUPPORTED_DATASETS. Please check the dataset name.")
    #         print(f"Supported datasets are: {const.SUPPORTED_DATASETS}")
    #         raise NotImplementedError(f"dataset '{params['dataset']}' is not supported.")
        
    #     if params['skip'] and params['config'] is None: continue

    #     all_folds = range(1, params['num_folds'] + 1)
    #     set_random_seed(param['seed'])
        
    #     if param['hypertune'] and not param['just_gen_dataset']: 
    #         EXP = 'perform_hypertunning' 
    #         assert param['combine_views_preds'] == 0, "Hypertuning is not supported with --combine_views_preds. Please set it to 0."
    #     elif not param['hypertune'] and not param['cross_dataset_test'] and not param['just_gen_dataset']: 
    #         EXP = 'perform_intra_dataset_test'
    #     elif not param['hypertune'] and param['cross_dataset_test'] and not params['LODO'] and not param['just_gen_dataset']:
    #         EXP = 'perform_cross_dataset_test'     
    #     elif not param['hypertune'] and param['cross_dataset_test'] and params['LODO'] and not param['just_gen_dataset'] and not param['AID']:
    #         EXP = 'perform_cross_dataset_test_LODO'
    #     elif not param['hypertune'] and param['cross_dataset_test'] and params['LODO'] and not param['just_gen_dataset'] and param['AID']:
    #         EXP = 'perform_cross_dataset_test_AID'
    main(params)