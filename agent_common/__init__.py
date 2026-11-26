"""Shared model runtime, model catalog, retry policy and recovery policy.

The project directory is ``agent-common`` (hyphenated, as Python project
names may be); this import package is ``agent_common`` (underscored, as
Python import names must be).  Consumers put the ``agent-common`` directory
on ``sys.path`` and then import ``agent_common.model_runtime`` or
``agent_common.recovery``.
"""
