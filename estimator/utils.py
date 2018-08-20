import collections
import six

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config


class RunConfig(tf.contrib.learn.RunConfig):
    def uid(self, whitelist=None):
        """Generate a 'Unique Identifier'
        """
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items()
                 if not k.startswith('__')}
        for k in whitelist:
            state.pop('_' + k, None)

        sorted_state = sorted(state.items(), key=lambda t: t[0])
        ordered_state = collections.OrderedDict(state)

        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                    sorted(ordered_state['_cluster_spec'].as_dict().items(),
                        key=lambda t: t[0]))
        return ', '.join('%s=%r' % (k, v) for (k, v) in
                six.iteritems(ordered_state))


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None):
        if (every_n_steps is None) == (every_n_secs is None):
            msg = 'exactly one of every_n_step and every_n_secs should be \
                provided'
            raise ValueError(msg)
        self._time = basic_session_run_hooks.SecondOrStepTimer(
                every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            msg = 'Global step should be created to use StepCounterHook'
            raise RuntimeError(msg)

    def before_run(self, run_context):
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._time.should_trigger_for_step(global_step):
            time, steps = self._time.update_last_triggered_step(global_step)
            if time is not None:
                steps_per_sec = steps / time
                self._step_train_time += time
                self._total_steps += steps

                average_examples_per_sec = self._batch_size * (
                        self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                        average_examples_per_sec, current_examples_per_sec,
                        self._total_steps)

def local_device_setter(num_devices=1, ps_device_type='cpu',
                        worker_device='/cpu:0', ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")
        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if nore_def.op in ps_ops:
            string = '/{}:{}'.format(ps_device_type, ps_strategy(op))
            ps_device_spec = pydev.DeviceSpec.from_string(string)
            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_dev_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_dev_spec.merge_from(current_device)
            return worker_dev_spec.to_string()
        return _local_device_chooser





