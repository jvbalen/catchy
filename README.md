## CATCHY

### Corpus Analysis Tools for Computational Hook discovery

Python tools for the corpus analysis of popular music recordings. The tools can be used separately or together. I.e.: you can use your own psychoacoustic features and still use the other modules. Note that to use all scripts, it is assumed that audio files come pre-segmented (e.g., into structural sections).

The base feature modules' requirements include Matlab, Librosa and VAMP.

### Structure

Extracting catchy features from a folder of files involves three steps:

1. Base feature extraction

Here, basic, familiar feature time series are extracted. The toolbox currently implements (wrappers for) MFCC, chroma, melody and perceptual feature extraction. (Rhythm features under development in branch `rhythm`.)
This part of the toolbox relies on a lot of external code, but it's also easy to work around: if you want to use other features, just save them to a set of csv files (1 per song section--see below) in some folder (1 per feature).

2. Pitch (and rhythm) descriptor extraction

This part computes mid-level pitch descriptors from chroma and/or melody information computed in step one. Essentially an implementation of several kinds of audio bigram descriptors. See also [1] and [2].

3. Feature transforms

Compute 'first' and 'second order' aggregates of any of the features computed in step 1 and step 2. See [2].

The above three steps correspond to the three columns in below diagram.

![Module Diagram](https://github.com/jvbalen/catchy/blob/master/catchy%20modules.png)

### Known issues:

- i/o currently very conservative--you may have to do your own mkdirs when writing features.

- Matlab path handling hasn't been checked on other machines than mine.
Hopefully these will be addressed soon.

### License

Matlab scripts under GNU Public license; everything else, see LICENSE.

If you use this, feel free to refer to [2].

[1] Van Balen, J., Wiering, F., & Veltkamp, R. (2015). Audio Bigrams as a Unifying Model of Pitch-based Song Description. In Proc. 11th International Symposium on Computer Music Multidisciplinary Research (CMMR). Plymouth, United Kingdom.

[2] Van Balen, J., Burgoyne, J. A., Bountouridis, D., Müllensiefen, D., & Veltkamp, R. (2015). Corpus Analysis Tools for Computational Hook Discovery. In Proc. 16th International Society for Music Information Retrieval Conference (pp. 227–233). Malaga, Spain.

Home page: [http://www.github.com/jvbalen/catchy](http://www.github.com/jvbalen/catchy)
(C) 2016 Jan Van Balen ([@jvanbalen](https://twitter.com/jvanbalen))