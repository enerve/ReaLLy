'''
Created on Apr 24, 2019

@author: enerve
'''

import logging
import numpy as np
import random
import torch

from really.epoch_trainer import EpochTrainer
from really.agent import *
# from function import BoundActionModel
# from function import NNBoundFA
from really.function import *
from really import cmd_line
from really import log
from really import util

from coindrop.lookahead_ab_agent import LookaheadABAgent
import coindrop.trainer_helper as th
from coindrop.episode_factory import EpisodeFactory
from coindrop.evaluator import Evaluator

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'Coindrop2'

    logger = logging.getLogger()
    log.configure_logger(logger, "Coindrop2")
    logger.setLevel(logging.DEBUG)
    
    # -------------- Configure track
    
    config = th.CONFIG(
        NUM_ROWS = 6,
        NUM_COLUMNS = 7
    )

    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    coindrop_fe = CoindropFeatureEng(config)
    bound_action_model = BoundActionModel(config)
    fe_elevation = FEElevation(config)
    
    NUM_NEW_EPISODES = 1000
    NUM_EPOCHS = 10
    
    logger.debug("NUM_NEW_EPISODES=%d\t NUM_EPOCHS=%d", NUM_NEW_EPISODES, NUM_EPOCHS)
    
    episode_factory = EpisodeFactory()

#     agent_fa = NN_FA(
#                     0.0005, # alpha ... #4e-5 old alpha without batching
#                     0, # regularization constant
#                     512, # batch_size
#                     500, # max_iterations
#                     coindrop_fe)
    agent_fa = NN_Bound_FA(
                    0.002, # alpha ... #4e-5 old alpha without batching
                    0.001, # regularization constant
                    512, # batch_size
                    20000, # max_iterations
                    fe_elevation)

    training_data_collector = FADataCollector(agent_fa)
    validation_data_collector = FADataCollector(agent_fa)

    es = ESPatches(config,
                  explorate=50000,
                  fa=agent_fa)
    explorer = FAExplorer(config, es)
    
    learner = th.create_agent(config, 
                    alg = 'sarsalambda',
                    lam = 0.95,
                    fa=agent_fa)
    
    # ------------------ Training -------------------

    opponent = LookaheadABAgent(config, 4)
    #opponent = FAExplorer(config, es)
    test_agent = FAAgent(config, agent_fa)

    evaluator = Evaluator(test_agent, opponent, 10)

    if False: # to train/test without exploration and processing
        util.pre_agent_alg = agent_fa.prefix()

        if False: # to train new model on dataset
            dir = "791563_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i30000_FFEelv__NNconvnet_lookab5__"
            agent_fa.initialize_default_net()
            training_data_collector.load_dataset(dir, "final_t")
            validation_data_collector.load_dataset(dir, "final_v")
            agent_fa.train(training_data_collector, validation_data_collector)
            agent_fa.report_stats()
        elif False: # to train existing model on dataset
            dir = "543562_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.0005_r0.5_b512_i1500_FFEelv__NNconvnet_lookab5__"
            agent_fa.load_model(dir, "v3")
            training_data_collector.load_dataset(dir, "final_t")
            validation_data_collector.load_dataset(dir, "final_v")
            agent_fa.train(training_data_collector, validation_data_collector)
            agent_fa.report_stats()
        elif True: #If load model params
            agent_fa.initialize_default_net()
            dir = "569160_Coindrop_DR_neural_bound_a0.002_r0.01_b512_i400_FFEelv__NNconvnet__"
            agent_fa.load_model_params(dir, "v3")
        elif False: #If load model architecture (and classes)
            dir = "569160_Coindrop_DR_neural_bound_a0.002_r0.01_b512_i400_FFEelv__NNconvnet__"
            agent_fa.load_model(dir, "v3")
            
        #agent_fa.save_model("v3")
        evaluator.run_test(10)
    
    elif True: # If Run episodes
        
        trainer = EpochTrainer(episode_factory, [explorer, opponent], learner, 
                               training_data_collector,
                               validation_data_collector,
                               evaluator,
                               explorer.prefix() + "_" + learner.prefix() + 
                               "_" + opponent.prefix())
        
        if True:
            # To start training afresh 
            agent_fa.initialize_default_net()
        elif False:
            # To start fresh but using existing episode history / exploration
            dir = "791563_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i30000_FFEelv__NNconvnet_lookab5__"
            agent_fa.initialize_default_net()
            explorer.load_episode_history("agent", dir)
            es.load_exploration_state(dir)
            opponent.load_episode_history("opponent", dir)
        elif False:
            # To start training from where we last left off.
            # i.e., load episodes history, exploration state, and FA model
            #dir = "831321_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i25000_FFEelv__NNconvnet_lookab5__"
            #dir = "789016_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i8000_FFEelv__NNconvnet_lookab5__"
            dir = "178719_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.001_b512_i150000_FFEelv__NNconvnet_lookab5__"
            explorer.load_episode_history("agent", dir)
            es.load_exploration_state(dir)
            opponent.load_episode_history("opponent", dir)
            agent_fa.load_model(dir, "v3")
            #trainer.load_stats(dir)
        elif False:
            # For single-epoch training/testing.
            # Load last training dataset and model, but not earlier history
            dir = "330041_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0005_r0_b512_i1000_F_NNconvnetlook3__"
            training_data_collector.load_dataset(dir, "final_t")
            validation_data_collector.load_dataset(dir, "final_v")
            agent_fa.load_model(dir, "v3")
    
        trainer.train(NUM_NEW_EPISODES, NUM_EPOCHS, 1)
        #trainer.save_to_file()

        explorer.store_episode_history("agent")
        es.store_exploration_state()
        opponent.store_episode_history("opponent")
        training_data_collector.store_last_dataset("final_t")
        validation_data_collector.store_last_dataset("final_v")
        agent_fa.save_model("v3")

        trainer.report_stats()
        trainer.save_stats()
    else: # load model, export to onnx
        
#         import onnx
#         model = onnx.load("/home/erwin/MLData/RL/output/655236_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0005_r0_b512_i1000_FFEv2__NNconvnetlook3__/coindropV2.onnx")
#         onnx.checker.check_model(model)
#         print(onnx.helper.printable_graph(model.graph))
        
        dir = "986337_Coindrop_DR_sarsa_lambda_eesp_l0.95neural_bound_a0.0005_r0.5_b512_i500_FBAM__NNconvnetlookab5__"
        agent_fa.load_model(dir, "v3")
        
        agent_fa.export_to_onnx("v3")
        #agent_fa.viz()
        
    
if __name__ == '__main__':
    main()

