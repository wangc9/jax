# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi process GPU tests on the same machine."""

import os
import subprocess
import sys

from absl import flags
import portpicker

from jax._src import test_util as jtu
from absl.testing import absltest

from jax.config import config
config.parse_flags_with_absl()

flags.DEFINE_integer("num_gpus", 4, "Number of GPUs.")
flags.DEFINE_integer("num_gpus_per_task", 1, "Number of gpus per task.")

FLAGS = flags.FLAGS

class MultiprocessGpuTest(jtu.JaxTestCase):

  def test(self):
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    port = portpicker.pick_unused_port()
    num_gpus = FLAGS.num_gpus
    num_gpus_per_task = FLAGS.num_gpus_per_task
    num_tasks = num_gpus // num_gpus_per_task

    os.environ["JAX_PORT"] = str(port)
    os.environ["NUM_TASKS"] = str(num_tasks)

    subprocesses = []
    for task in range(num_tasks):
      env = os.environ.copy()
      env["TASK"] = str(task)
      env["CUDA_VISIBLE_DEVICES"] = ",".join(
          str((task * num_gpus_per_task) + i) for i in range(num_gpus_per_task))
      args = [
          f"{py_version}",
          "-c",
          '"import jax, os; jax.distributed.initialize(f\"localhost:{os.environ[\"JAX_PORT\"]}\", os.environ[\"NUM_TASKS\"], os.environ[\"TASK\"])"'
      ]
      subprocesses.append(subprocess.Popen(args, env=env, shell=True))

    for i in range(num_tasks):
      self.assertEqual(subprocesses[i].wait(), 0)


if __name__ == "__main__":
  absltest.main()
