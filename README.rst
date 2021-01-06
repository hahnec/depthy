======
depthy
======

Description
-----------

*depthy* enables depth map generation from light-fields and stereo image pairs.

|release| |license| |build_github| |coverage| |pypi_total| |pypi|

Light-field Results
-------------------

|vspace|

.. list-table::
   :widths: 8 8 8 8
   :header-rows: 0
   :stub-columns: 1

   * - Central view
     - |photo_ref_1|
     - |photo_ref_2|
     - |photo_ref_3|
   * - Depth map
     - |depth_map_1|
     - |depth_map_2|
     - |depth_map_3|

|

Stereo Results
--------------

|vspace|

.. list-table::
   :widths: 8 8 8 8
   :header-rows: 1
   :stub-columns: 0

   * - Left Image
     - Ground Truth
     - Depthy (SGM)
     - OpenCV
   * - |photo_left_1|
     - |depth_gtd_1|
     - |depth_sgm_1|
     - |depth_ocv_1|

|

Installation
------------

* via pip:
    1. install with ``pip3 install depthy``
    2. type ``depthy -h`` to the command line once installation finished

* from source:
    1. install Python from https://www.python.org/
    2. download the source_ using ``git clone https://github.com/hahnec/depthy.git``
    3. go to the root directory ``cd depthy``
    4. load dependencies ``$ pip3 install -r requirements.txt``
    5. install with ``python3 setup.py install``
    6. if installation ran smoothly, enter ``depthy -h`` to the command line

Command Line Usage
------------------

From the root directory of your downloaded repo, you can run the tool on the provided test data by

``depthy -s './examples/data/pens/'``

on a UNIX system where the result is found at ``./examples/data/``. A windows equivalent of the above command is

``depthy --src=".\\examples\\data\\pens\\"``

Alternatively, you can specify the method or select your images manually with

``depthy --win --method='epi'``

More information on optional arguments, can be found using the help parameter

``depthy -h``

API Usage
---------

A documentation on the key functions is found at `https://hahnec.github.io/depthy/ <https://hahnec.github.io/depthy/>`__.
A reference integration using depthy's API can be found in `PlenoptiCam <https://github.com/hahnec/plenopticam/blob/master/plenopticam/lfp_extractor/lfp_depth.py>`__.

Author
------

`Christopher Hahne <http://www.christopherhahne.de/>`__

Citation
--------

.. code-block:: BibTeX

    @misc{depthy,
        title={Depthy},
        author={Christopher Hahne},
        year={2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/hahnec/depthy}}
    }

.. Hyperlink aliases

.. _source: https://github.com/hahnec/depthy/archive/master.zip

.. |photo_ref_1| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/pens_040.png" width="200px" max-width:"100%">

.. |photo_ref_2| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/herbs_040.png" width="200px" max-width:"100%">

.. |photo_ref_3| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/boxes_040.png" width="200px" max-width:"100%">

.. |depth_map_1| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/pens.png" width="200px" max-width:"100%">

.. |depth_map_2| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/herbs.png" width="200px" max-width:"100%">

.. |depth_map_3| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/boxes.png" width="200px" max-width:"100%">

.. |photo_left_1| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/im6.png" width="200px" max-width:"100%">

.. |depth_gtd_1| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/disp6.png" width="200px" max-width:"100%">

.. |depth_sgm_1| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/cones_sgm_l.png" width="200px" max-width:"100%">

.. |depth_ocv_1| raw:: html

    <img src="https://raw.githubusercontent.com/hahnec/depthy/master/docs/img/cones_ocv_l.png" width="200px" max-width:"100%">


.. |vspace| raw:: latex

   \vspace{1mm}

.. Image substitutions

.. |release| image:: https://img.shields.io/github/v/release/hahnec/depthy?style=square
    :target: https://github.com/hahnec/depthy/releases/
    :alt: Release

.. |license| image:: https://img.shields.io/badge/License-GPL%20v3.0-orange.svg?style=square
    :target: https://www.gnu.org/licenses/gpl-3.0.en.html
    :alt: License

.. |build_github| image:: https://img.shields.io/github/workflow/status/hahnec/depthy/Depthy's%20CI%20Pipeline/develop?style=square
    :target: https://github.com/hahnec/depthy/actions
    :alt: GitHub Workflow Status

.. |build_travis| image:: https://img.shields.io/travis/com/hahnec/depthy?style=square
    :target: https://travis-ci.com/github/hahnec/depthy
    :alt: Travis CI Status

.. |coverage| image:: https://img.shields.io/coveralls/github/hahnec/depthy?style=square
    :target: https://coveralls.io/github/hahnec/depthy
    :alt: Coveralls Status

.. |pypi| image:: https://img.shields.io/pypi/dm/depthy?label=PyPI%20downloads&style=square
    :target: https://pypi.org/project/depthy/
    :alt: PyPI Downloads

.. |pypi_total| image:: https://pepy.tech/badge/depthy?style=flat-square
    :target: https://pepy.tech/project/depthy
    :alt: PyPi Dl2