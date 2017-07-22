tf_G
====


:Name: tf_G
:Description: Python's Tensorflow Graph Library
:Website: https://github.com/garciparedes/tf_G
:Author: `@garciparedes <http://garciparedes.me>`__
:Version: 0.1

.. |travisci| image:: https://img.shields.io/travis/AeroPython/PyFME/master.svg?style=flat-square
   :target: https://travis-ci.org/garciparedes/tf_G

.. |codecov| image:: https://img.shields.io/codecov/c/github/garciparedes/tf_G.svg?style=flat-square
   :target: https://codecov.io/gh/garciparedes/tf_G?branch=master

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square
   :target: http://tf_G.readthedocs.io/en/latest/?badge=latest

.. |gitter| image:: https://badges.gitter.im/tf_G/Lobby.svg
   :alt: Join the chat at https://gitter.im/tf_G/Lobby
   :target: https://gitter.im/tf_G/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

|travisci| |codecov| |docs| |gitter|

Description
--------------------------------------------------------------------------------
This work consists of a study of a set of techniques and strategies related with algorithm's design, whose purpose is the resolution of problems on massive data sets, in an efficient way. This field is known as Algorithms for Big Data. In particular, this work has studied the Streaming Algorithms, which represents the basis of the data structures of sublinear order o(n) in space, known as Sketches. In addition, it has deepened in the study of problems applied to Graphs on the Semi-Streaming model. Next, the PageRank algorithm was analyzed as a concrete case study. Finally, the development of a library for the resolution of graph problems, implemented on the top of the intensive mathematical computation platform known as TensorFlow has been started.

Content
-------
* `Source Code <https://github.com/garciparedes/tf_G/blob/master/src/tf_G>`__
* `API Documentation <http://tf-g.readthedocs.io/>`__
* `Code Examples <https://github.com/garciparedes/tf_G/blob/master/examples>`__
* `Tests <https://github.com/garciparedes/tf_G/blob/master/tests>`__
* `Final Degree Project: Memory <https://github.com/garciparedes/tf_G/blob/master/tex/document/document.pdf>`__
* `Final Degree Project: Slides <https://github.com/garciparedes/tf_G/blob/master/tex/slides/slides.pdf>`__
* `Final Degree Project: Summary <https://github.com/garciparedes/tf_G/blob/master/tex/summary/summary.pdf>`__


How to install
--------------

If you have git installed, you can try::

    $ pip install git+https://github.com/garciparedes/tf_G.git

If you get any installation or compilation errors, make sure you have the latest pip and setuptools::

    $ pip install --upgrade pip setuptools

How to run the tests
--------------------

Install in editable mode and call `pytest`::

    $ pip install -e .
    $ pytest
