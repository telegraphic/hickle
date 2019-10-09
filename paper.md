---
title: 'Hickle: A HDF5-based python pickle replacement'
tags:
  - Python
  - astronomy
authors:
  - name: Danny C. Price
    orcid: 0000-0003-2783-1608
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Ellert van der Velden
    orcid: 0000-0002-1559-9832  
    affiliation: 2
  - name: Sébastien Celles
    orcid: 0000-0001-9987-4338
    affiliation: 3
  - name: Pieter T. Eendebak
    orcid: 0000-0001-7018-1124
    affiliation: "4, 5"
  - name: Michael M. McKerns
    orcid: 0000-0001-8342-3778
    affiliation: 6
  - name: Eben M. Olson
    affiliation: 7
  - name: Colin Raffel
    affiliation: 8
  - name: Bairen Yi
    affiliation: 9
  - name: Elliott Ash
    affiliation: 10
affiliations:
  - name: Department of Astronomy,  University of California Berkeley, Berkeley CA 94720
    index: 1
  - name: Centre for Astrophysics & Supercomputing, Swinburne University of Technology, Hawthorn, VIC 3122, Australia
    index: 2
  - name: Thermal Science and Energy Department, Institut Universitaire de Technologie de Poitiers - Université de Poitiers, France
    index: 3
  - name: QuTech, Delft University of Technology, P.O. Box 5046, 2600 GA Delft, The Netherlands
    index: 4
  - name: Netherlands Organisation for Applied Scientific Research (TNO), P.O. Box 155, 2600 AD Delft, The Netherlands
    index: 5
  - name: Institute for Advanced Computational Science, Stony Brook University, Stony Brook, NY 11794-5250
    index: 6
  - name: Department of Laboratory Medicine, Yale University, New Haven CT 06510 USA
    index: 7
  - name: Google Brain, Mountain View, CA, 94043
    index: 8
  - name: The Hong Kong University of Science and Technology
    index: 9
  - name: ETH Zurich
    index: 10
date: 10 November 2018
bibliography: paper.bib
---

# Summary
``hickle`` is a Python 2/3 package for quickly dumping and loading python data structures to Hierarchical Data Format 5 (HDF5) files [@hdf5]. When dumping to HDF5, ``hickle`` automatically convert Python data structures (e.g. lists, dictionaries, ``numpy`` arrays [@numpy]) into HDF5 groups and datasets. When loading from file, ``hickle`` automatically converts data back into its original data type. A key motivation for ``hickle`` is to provide high-performance loading and storage of scientific data in the widely-supported HDF5 format.

``hickle`` is designed as a drop-in replacement for the Python ``pickle`` package, which converts Python object hierarchies to and from Python-specific byte streams (processes known as 'pickling' and 'unpickling' respectively). Several different protocols exist, and files are not designed to be compatible between Python versions, nor interpretable in other languages. In contrast, ``hickle`` stores and loads files from HDF5, for which application programming interfaces (APIs) exist in most major languages, including C, Java, R, and MATLAB.

Python data structures are mapped into the HDF5 abstract data model in a logical fashion, using the ``h5py`` package [@collette:2014]. Metadata required to reconstruct the hierarchy of objects, and to allow conversion into Python objects, is stored in HDF5 attributes. Most commonly used Python iterables (dict, tuple, list, set), and data types (int, float, str) are supported, as are ``numpy``  N-dimensional arrays. Commonly-used ``astropy`` data structures and ``scipy`` sparse matrices are also supported.

``hickle`` has been used in many scientific research projects, including:

* Visualization and machine learning on volumetric fluorescence microscopy datasets from histological tissue imaging [@Durant:2017].
* Caching pre-computed features for MIDI and audio files for downstream machine learning tasks [@Raffel:2016].  
* Storage and transmission of high volume of shot-gun proteomics data, such as mass spectra of proteins and peptide segments [@Zhang:2016].
* Storage of astronomical data and calibration data from radio telescopes [@Price:2018].

``hickle`` is released under the MIT license, and is available from PyPi via ``pip``; source code is available at
 https://github.com/telegraphic/hickle.

# References
