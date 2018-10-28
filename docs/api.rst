.. _api:

API Documentation
=================

Getting Started
---------------

To parse text to UCCA passages, download a model file from `the latest release <https://github.com/huji-nlp/tupa/releases>`__, extract it, and use the following code template::

    from tupa.parse import Parser
    from ucca.convert import from_text
    parser = Parser("models/ucca_bilstm")
    for passage in parser.parse(from_text(...))):
        ...

Each passage instantiates the `ucca.core.Passage <https://ucca.readthedocs.io/en/latest/api/ucca.core.Passage.html#ucca.core.Passage>`__ class.

.. automodapi:: tupa.parse
.. automodapi:: tupa.action
.. automodapi:: tupa.config
.. automodapi:: tupa.labels
.. automodapi:: tupa.model
.. automodapi:: tupa.model_util
.. automodapi:: tupa.oracle

