'''
Created on Sep 17, 2019

@author: enerve
'''

import logging
from really import util

import numpy as np
import torch

class PolicyLearner:
    '''
    Observer that generates policy training data from episode-history
    '''

    def __init__(self,
                 pa,
                 fa):
        
        self.pa = pa
        self.fa = fa

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.max_moves = 100
        
        # stats
        self.stats_abs_delta = {}
        # for histograms
        self.stats_deltas = {}
        self.stats_currs = {}
        self.stats_targets = {}

    def prefix(self):
        return "pal"

    def save_stats(self, pref=""):
        self.pa.save_stats(pref)
        
        util.dump(self.stats_abs_delta, "statsDelta", pref)

    def load_stats(self, subdir, pref=""):
        self.pa.load_stats(subdir, pref)
        
        self.stats_abs_delta = util.load("statsDelta", subdir, pref)[()]

    def report_stats(self, pref):
        self.pa.report_stats(pref)
        
        delta_list = list(self.stats_abs_delta.values())
        labels = list(self.stats_abs_delta.keys()) 
        util.plot(delta_list,
                  range(len(delta_list[0])),
                  labels=labels,
                  title="Average Δ",
                  pref='delta')
        
        
    def live_stats(self):
        #self.pa.live_stats()
 
        pass

    # ----------- Analyze given history to extract training/val data -----------

    def process(self, episodes_history, data_collector, source_name):
        ''' Analyzes given history to create data rows from it '''
        # TODO: remove / changes stats? perhaps just magnitude of gradients
        self.deltas = []
        self.targets = []
        self.currs = []

        #self.num_wins = 0
        #self.num_loss = 0
        
        self.logger.debug(" Process history of %d episodes", len(episodes_history))

        for i, steps_history in enumerate(episodes_history):
            self._process_steps(steps_history, data_collector)

        # stats
        npd = np.array(self.deltas)
        if source_name not in self.stats_abs_delta:
            self.stats_abs_delta[source_name] = []
        if source_name not in self.stats_deltas:
            self.stats_deltas[source_name] = []
            self.stats_currs[source_name] = []
            self.stats_targets[source_name] = []
        self.stats_abs_delta[source_name].append(np.mean(np.abs(npd)))
        self.stats_deltas[source_name].append(self.deltas)
        self.stats_currs[source_name].append(self.currs)
        self.stats_targets[source_name].append(self.targets)
        
#         self.logger.debug("Currs:\n%s", self.currs[0:200])
#         self.logger.debug("Targets:\n%s", self.targets[0:200])
#         self.logger.debug("CurrMean:   %f", np.array(self.currs).mean())
#         self.logger.debug("TargetMean: %f", np.array(self.targets).mean())
        
        
        #self.logger.debug('  sumdelta: %0.4f', self.sum_delta)
        self.logger.debug('  delta   : %0.2f <%0.2f>', np.mean(np.abs(npd)), np.var(npd))
        #self.logger.debug('  wins/losses (total): %d/%d (%d)', self.num_wins,
        #                  self.num_loss, len(episodes_history))

    def _process_steps(self, steps_history, data_collector):
        ''' Observes and learns from the given episode '''
        
        S, A = None, None
        for i, (R_, S_, A_) in enumerate(steps_history):
            if i > 0:
                # Compute score function of softmax policy
                #enc_vecs = self.pa.encoding_vectors(S)
                _, activations, valid_actions = self.pa.values(S)
                enc_vecs = activations['features']
                
                ai = valid_actions.index(A)
                mean_vecs = torch.mean(enc_vecs, axis=0)
                scorefn = enc_vecs[ai] - mean_vecs
                
                fa_value = self.fa.value(S, A)

                gradient = scorefn * fa_value

                data_collector.record(S, A, gradient)

            S, A = S_, A_
        
        
    def plot_last_hists(self, source_name):
        # TODO: maybe move this to DataCollector instead
        
        npc = np.array(self.stats_currs[source_name][-1])
        util.hist(npc, 100, (-2, 2),
                  "%s curr value" % source_name,
                  "currhist" + source_name)

        npt = np.array(self.stats_targets[source_name][-1])
        util.hist(npt, 100, (-2, 2),
                  "%s targets" % source_name,
                  "targetshis_t" + source_name)

        npd = np.array(self.stats_deltas[source_name][-1])
        util.hist(npd, 100, (-2, 2),
                  "%s delta" % source_name,
                  "deltahist_" + source_name)

    def write_hist_animation(self, source_name):
        # TODO: maybe move this to DataCollector instead

        maxlen = len(self.stats_currs[source_name][0])
        # TODO: Thsi is racecar-specific.
        rng_min = -400 #-1.2
        rng_max = 400  #1.2
        
        util.save_hist_animation(self.stats_currs[source_name], 100, (rng_min, rng_max),
                                 maxlen, "curr value", "currhist")
        util.save_hist_animation(self.stats_targets[source_name], 100, (rng_min, rng_max),
                                 maxlen, "targets", "targethist")
        util.save_hist_animation(self.stats_deltas[source_name], 100, (rng_min, rng_max),
                                 maxlen, "delta", "deltahist")

    def save_hists(self, sources):
        currs = []
        targets = []
        deltas = []
        for source in sources:
            currs.extend(self.stats_currs[source])
            targets.extend(self.stats_targets[source])
            deltas.extend(self.stats_deltas[source])
            
        A = np.asarray([currs,
                        targets,
                        deltas])
        util.dump(A, "valuehistory")

    def load_hists(self, subdir):
        # TODO: figure out what exactly we want to store/load
        A = util.load("valuehistory", subdir).tolist()
        self.all_currs, self.all_targets, self.all_deltas = A[0], A[1], A[2]

    # ---------------- Train ---------------

    def learn(self, data_collector, validation_data_collector):
        ''' Process the training data collected since last update.
        '''
        self.pa.update(data_collector, validation_data_collector)

    def save_model(self, pref=""):
        self.pa.save_model(pref)

    def load_model(self, load_subdir, pref=""):
        self.pa.load_model(load_subdir, pref)
