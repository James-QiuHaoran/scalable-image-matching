Getting started
===============
You'll need a (scientific) Python distribution and a database backend.
Currently we use Elasticsearch as a backend.


numpy, PIL, skimage, etc.
-------------------------
``image_match`` requires several scientific Python packages. Although they can
be installed and built individually, they are often bundled in a custom Python
distribution, for instance `Anaconda`_. Installation instructions can be found
`here <https://www.continuum.io/downloads#_unix>`_.

You can set up ``image_match`` without a prebuilt distribution, but the
performance may suffer. Note that ``scipy`` and ``numpy`` require many
system-level dependencies that you made need to install first.


Elasticsearch
-------------
If you just want to generate and compare image signatures, you can skip this
step. If you want to search over a corpus of millions or billions of image
signatures, you will need a database backend. We built ``image_match`` around
`Elasticsearch`_. See `download and installation instructions <https://www.elastic.co/downloads/elasticsearch>`_.  We're using
``Elasticsearch 2.2.1`` in these examples.


Install image-match
-------------------
Here are a few options:

Install with pip
^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ pip install numpy
    $ pip install scipy
    $ pip install image_match

Build from source
^^^^^^^^^^^^^^^^^

1. Clone this repository:

    .. code-block:: bash
  
        $ git clone https://github.com/ascribe/image-match.git

2. Install ``image_match``

    From the project directory:
  
    .. code-block:: bash
    
        $ pip install numpy
        $ pip install scipy
        $ pip install .

Make sure elasticsearch is running (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For example, on Ubuntu you can check with:

.. code-block:: bash
    
    $ sudo service elasticsearch status


If it's not running, simply run:

.. code-block:: bash

    $ sudo service elasticsearch start


On OSX, to have ``launchd`` start elasticsearch, run : 

.. code-block:: bash

    $ brew services start elasticsearch

or simply run ,

.. code-block:: bash

    $ elasticsearch

Docker
^^^^^^
We have a ``Docker`` image that takes care of setting up ``image_match`` and
elasticsearch. Consider it an alternative to the methods described above.

.. code-block:: bash

    $ docker pull ascribe/image-match
    $ docker run -it ascribe/image-match /bin/bash


.. _Anaconda: https://www.continuum.io/why-anaconda
.. _Elasticsearch: https://www.elastic.co/
