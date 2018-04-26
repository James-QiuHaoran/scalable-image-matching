Storing and searching the Signatures
====================================
In addition to generating image signatures, ``image_match`` also facilitates
storing and efficient lookup of images—even for up to (at least) a billion
images.  Instagram account only has a few million images? Don't worry, you can
get 80M images `here <http://horatio.cs.nyu.edu/mit/tiny/data/index.html>`_ to
play with.

A signature database wraps an Elasticsearch index, so you'll need Elasticsearch
up and running. Once that's done, you can set it up like so:

.. code-block:: python

    from elasticsearch import Elasticsearch
    from image_match.elasticsearch_driver import SignatureES

    es = Elasticsearch()
    ses = SignatureES(es)


By default, the Elasticsearch index name is ``'images'`` and the document type
``'image'``, but you can change these via the ``index`` and ``doc_type``
parameters.

Now, let's store those pictures from before in the database:

.. code-block:: python

    ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
    ses.add_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
    ses.add_image('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
    ses.add_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')

Now let's search for one of those Mona Lisas:

.. code-block:: python

    ses.search_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')

The result is a list of hits:

.. code-block:: python

    [
     {'dist': 0.0,
      'id': u'AVM37oZq0osmmAxpPvx7',
      'metadata': None,
      'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
      'score': 7.937254},
     {'dist': 0.22095170140933634,
      'id': u'AVM37nMg0osmmAxpPvx6',
      'metadata': None,
      'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
      'score': 0.28797293},
     {'dist': 0.42557196987336648,
      'id': u'AVM37p530osmmAxpPvx9',
      'metadata': None,
      'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg',
      'score': 0.0499953}
    ]

``dist`` is the normalized distance, like we computed above. Hence, lower numbers
are better with ``0.0`` being a perfect match. ``id`` is an identifier assigned by
the database. ``score`` is computed by Elasticsearch, and higher numbers are
better here. ``path`` is the original path (url or file path). ``metadata`` is
an optional field used for storing extra information about the image (see below).

Notice all three Mona Lisa images appear in the results, with the identical
image being a perfect (``'dist': 0.0``) match. If we search instead for the
Caravaggio,

.. code-block:: python

    ses.search_image('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')

You get:

.. code-block:: python

    [
     {'dist': 0.0,
      'id': u'AVMyXQFw0osmmAxpPvxz',
      'metadata': None,
      'path': u'https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg',
      'score': 7.937254}
    ]

It only finds the Caravaggio, which makes sense! But what if we wanted an even
more restrictive search? For instance, maybe we only want unmodified Mona Lisas
-- just photographs of the original. We can restrict our search with a hard
cutoff using the ``distance_cutoff`` keyword argument:

.. code-block:: python

    ses = SignatureES(es, distance_cutoff=0.3)
    ses.search_image('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')

Which now returns only the unmodified, catless Mona Lisas:

.. code-block:: python

    [
     {'dist': 0.0,
      'id': u'AVMyXOz30osmmAxpPvxy',
      'metadata': None,
      'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
      'score': 7.937254},
     {'dist': 0.23889600350807427,
      'id': u'AVMyXMpV0osmmAxpPvxx',
      'metadata': None,
      'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
      'score': 0.28797293}
    ]

Distorted and transformed images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``image_match`` is also robust against basic image transforms. Take this
squashed Mona Lisa:

.. image:: http://i.imgur.com/CVYBCCy.jpg

No problem, just search as usual:

.. code-block:: python

    ses.search_image('http://i.imgur.com/CVYBCCy.jpg')

returns

.. code-block:: python

    [
     {'dist': 0.15454905655638429,
      'id': u'AVM37oZq0osmmAxpPvx7',
      'metadata': None,
      'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
      'score': 1.6818419},
     {'dist': 0.24980626832071956,
      'id': u'AVM37nMg0osmmAxpPvx6',
      'metadata': None,
      'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
      'score': 0.16198477},
     {'dist': 0.43387141782958921,
      'id': u'AVM37p530osmmAxpPvx9',
      'metadata': None,
      'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg',
      'score': 0.031996995}
    ]

as expected.  Now, consider this rotated version:

.. image:: http://i.imgur.com/T5AusYd.jpg

``image_match`` doesn't search for rotations and mirror images by default.
Searching for this image will return no results, unless you search with
``all_orientations=True``:

.. code-block:: python

    ses.search_image('http://i.imgur.com/T5AusYd.jpg', all_orientations=True)

Then you get the expected matches.

Adding metadata
^^^^^^^^^^^^^^^
Sometimes you want to store information with your images independent of the
reverse image search functionality.  You can do that with the ``metadata=``
field in the ``add_image`` function.

Let's add one of the images again, with some extra data:

.. code-block:: python

    ses.add_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg', metadata={'things': 'stuff!'})

In general, any JSON-like data should work with ``metadata=``. Now we can search for the image:

.. code-block:: python

    ses.search_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')

Returns our previous results along with a new one:

.. code-block:: python

    [
     {'dist': 0.0,
      'id': u'AVYhQYhEDpLcdyATKuy-',
      'metadata': None,
      'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg',
      'score': 7.64685},
     {'dist': 0.0,
      'id': u'AVYhRvoWDpLcdyATKuzE',
      'metadata': {u'things': u'stuff!'},
      'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg',
      'score': 2.435569},
      ...
     ]

Where we can see a little extra info. ``image-match`` doesn't provide anyway to query
the metadata directly, but the user can use Elasticsearch's QL, for example with:

.. code-block:: python

    ses.es.search('images', body={'query': {'match': {'metadata.things': 'stuff!'}}})

