import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import attacks.DBA.train as train_dba
import attacks.DBA.test as test_dba
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms

from attacks.DBA.image_helper import ImageHelper
from attacks.DBA.utils.utils import dict_html
import attacks.DBA.utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config 
import copy
import defenses.pca_deflect

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")



criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def trigger_test_byindex(helper, index, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_dba.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])

def trigger_test_byname(helper, agent_name_key, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test_dba.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    


def main_dba(defense_name, dataset_name):
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    # parser = argparse.ArgumentParser(description='PPDL')
    # parser.add_argument('--params', dest='params')
    # args = parser.parse_args()
    # with open(f'./{args.params}', 'r') as f:
    #     params_loaded = yaml.safe_load(f)
    with open('./attacks/DBA/utils/cifar_params.yaml') as f:
        params_loaded = yaml.safe_load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    
    params_loaded['aggregation_methods'] = defense_name
    params_loaded['type'] = dataset_name
    if params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    else:
        helper = None

    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')

    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  # must have advasarial if this epoch is in their poison epoch
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])

                nonattacker=[]
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist+nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary']==False:
                adversarial_name_keys=copy.deepcopy(helper.params['adversary_list'])
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        epochs_submit_update_dict, num_samples_dict = train_dba.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        logger.info(f'time spent on training: {time.time() - t}')
        if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT:
            client_names   = list(epochs_submit_update_dict.keys())
            client_weights = [updates[-1] for updates in epochs_submit_update_dict.values()]

            client_weights_flat = defenses.pca_deflect.extract_client_weights(client_weights)
            

            outliers, _ = defenses.pca_deflect.apply_pca_to_weights(client_weights_flat, client_names, epoch, [])
            print(outliers)

            for bad in outliers:
                epochs_submit_update_dict.pop(bad, None)   # drop their weight updates :contentReference[oaicite:0]{index=0}
                num_samples_dict.pop(bad,         None)   # drop their sample counts  :contentReference[oaicite:1]{index=1}
                if bad in agent_name_keys:
                    agent_name_keys.remove(bad)  
                    print("Removed Outlier's influence", bad) 

        agents_num = len(agent_name_keys)

         
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        
        if helper.params["aggregation_methods"] == config.AGGR_PCA_DEFLECT:
            for bad in outliers:
                if bad not in agent_name_keys:
                    agent_name_keys.append(bad)


        is_updated = True
        if helper.params['aggregation_methods'] == config.AGGR_MEAN or helper.params['aggregation_methods'] == config.AGGR_PCA_DEFLECT:
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'], agents_num=agents_num)
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)

        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            num_oracle_calls = 1

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test_dba.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])


        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if helper.params['is_poison']:

            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test_dba.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])


            # test on local triggers
            csv_record.poisontriggertest_result.append(
                ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            logger.info(f"[Epoch {temp_global_epoch}] Backdoor Accuracy (combine): "
            f"{epoch_corret}/{epoch_total} ({epoch_acc_p:.2f}%)")
            
            if helper.params['vis_trigger_split_test']:
                helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=None,
                                                           eid=helper.params['environment_name'],
                                                           name="global_combine")
            if len(helper.params['adversary_list']) == 1:  # centralized attack
                if helper.params['centralized_test_trigger'] == True:  # centralized attack test on local triggers
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, epoch)
            else:  # distributed attack
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, epoch)
                    # logger.info(f"Backdoor Accuracy (combine) at epoch {temp_global_epoch}: "f"{epoch_corret}/{epoch_total} ({100.*epoch_acc_p:.2f}%)")


        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)



    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")



