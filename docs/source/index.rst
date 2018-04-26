***********
image-match
***********
image-match is a simple package for finding approximate image matches from a
corpus. It is similar, for instance, to `pHash <http://www.phash.org/>`_, but
includes a database backend that easily scales to billions of images and
supports sustained high rates of image insertion: up to 10,000 images/s on our
cluster!

Based on the paper `An image signature for any kind of image, Wong et al`_
There is an existing `reference implementation <https://www.pureftpd.org/project/libpuzzle>`_ which may be more suited to your needs.

The folks over at `Pavlov <https://pavlovml.com/>`_ have released an excellent
`containerized version of image-match <https://github.com/pavlovml/match>`_ for
easy scaling and deployment.

.. _An image signature for any kind of image, Wong et al: http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps

Contents
========

.. toctree::
    :maxdepth: 2
    
    start
    signatures
    searches
    backends
    docs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
