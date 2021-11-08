"""Core deep symbolic optimizer construct."""

import json
import zlib
from collections import defaultdict
from multiprocessing import Pool

import tensorflow as tf

from dsr.task import set_task
from dsr.controller import Controller
from dsr.train import learn
from dsr.prior import make_prior
from dsr.program import Program


class DeepSymbolicOptimizer():
    """
    Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.

    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON. See dsr/dsr/config.json for template.

    Attributes
    ----------
    config : dict
        Configuration parameters for training.

    Methods
    -------
    train
        Builds and trains the model according to config.
    """

    def __init__(self, config=None):
        self.update_config(config)
        self.sess = None

    def setup(self, seed=0):

        # Clear the cache, reset the compute graph, and set the seed
        Program.clear_cache()
        tf.reset_default_graph()
        self.seed(seed) # Must be called _after_ resetting graph

        self.pool = self.make_pool()
        self.prior = self.make_prior()

        if self.config_task['enforce_sum']:
            n_tensors = 0
            for input in self.config_task['dataset_info']['input']:
                # count the number of tensors in the inputs, if any:
                if input in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
                    n_tensors += 1
        else:
            n_tensors = 1

        self.sess = []
        self.controller = []

        for ii in range(n_tensors):
            graph = tf.Graph()
            with graph.as_default():
                self.seed(seed+ii)
                new_sess = tf.Session()
                self.sess.append(new_sess)
                new_controller = Controller(new_sess,
                                            self.prior,
                                            **self.config_controller)
                new_controller.sess.run(tf.global_variables_initializer())  # initializer should be part of the graph
                self.controller.append(new_controller)

    def train(self, seed=0):

        # Setup the model
        self.setup(seed)

        # Train the model
        result = learn(self.sess,
                       self.controller,
                       self.pool,
                       **self.config_training)
        return result

    def update_config(self, config):
        if config is None:
            config = {}
        elif isinstance(config, str):
            with open(config, 'rb') as f:
                config = json.load(f)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_training = self.config["training"]
        self.config_controller = self.config["controller"]

    def seed(self, seed_=0):
        """Set the tensorflow seed, which will be offset by a checksum on the
        task name to ensure seeds differ across different tasks."""

        if "name" in self.config_task:
            task_name = self.config_task["name"]
        else:
            task_name = ""
        seed_ += zlib.adler32(task_name.encode("utf-8"))
        tf.set_random_seed(seed_)

        return seed_

    def make_prior(self):
        prior = make_prior(Program.sec_library, self.config_prior)
        return prior

    def make_controller(self):
        controller = []
        for sess in self.sess:
            controller.append(Controller(sess,
                                         self.prior,
                                         **self.config_controller))
        return controller

    def make_pool(self):
        # Create the pool and set the Task for each worker
        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None and n_cores_batch > 1:
            pool = Pool(n_cores_batch,
                        initializer=set_task,
                        initargs=(self.config_task,))

        # Set the Task for the parent process
        set_task(self.config)

        return pool

    def save(self, save_path):

        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):

        if self.sess is None:
            self.setup()
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
