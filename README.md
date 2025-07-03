# diffrogram
The reference Matlab utility for DF measurements in audio.

## Overview
The function compares two audio files which usually represent reference and output signals of some device under test. It operates piece-wise with a specified time window and computes the array of difference levels and powers:

- *df(:,1)* - waveform DF (DFwf, dB)
- *df(:,2)* - magnitude DF (DFmg, dB)
- *df(:,3)* - phase DF (DFph, dB)
- *df(:,4)* - RMS power (PP, dB)

Each *df* vector is accompanied by the diffrogram – the image which has Time along horizontal dimension and Frequency - along vertical.
It shows degradation of an output signal with time and frequency. Color of each pixel corresponds to DF level of an output signal in
particular time frame and frequency sub-band. Brightness of its color corresponds to the energy of this portion of the output signal.
DF/color mapping is defined algorithmically (*colormap* option). On the diffrograms DF values below -150 dB are clipped to -150 dB; DF
value -Inf dB (perfect match between reference and output waveforms) is coded with Grey. File name of the diffrogram image includes
Median, Max and Min DF values of corresponding *df* vector. First and last values for a signal are not counted as they are often erroneous
due to edge effect.

For correct calculation of DF levels an output signal must be precicely time/pitch aligned to the reference signal. If required, this operation
of time warping can be performed by the diffrogram function too. Resulting *warped* output signal is returned by the function.

## Syntax and description
*df = diffrogram33(fref, fout, Tw, options)* reads the audio files named *fref* and *fout* and returns three output arguments:

- an array df of DFwf, DFmg and DFph values in dB,
- three corresponding diffrogram images and
- a warped output signal.

The audio files can be of any sample rate and bit depth. All DF levels are computed for the region of interest 20Hz - 20kHz regardless of
the sample rate.

*Tw* - width of the rectangular time window in milliseconds used for computing DF levels; if *Tw* is greater than the length of a reference
signal, the *diffrogram33* function computes scalar DFwf, DFmg and DFph levels for the whole signal.

*options* (a string, case insensitive, any order) must include at least one of the following arguments:

- *Left* | *Right* | *Mono* - channel mode. Default: *Mono*

- *NoWarp* - skip time warping of the output signal if it is known to be perfectly time/pitch aligned already.

- *SyncMargin:integer* - time frame in milliseconds at the beginning and the end of a reference signal that is used for cutting the
output signal accordingly. The default value is 3000 ms (or less if the signal is shorter). Increase the value if the signals contain
too quiet passages at the beginning and the end. Periodic signals of high frequency may require shorter *SyncMargin* values: *1-10*. The output signal for the function can be prepared (cut out) approximately, with some extra margin of about 1/10th of
*SyncMargin* value. If *0* is specified the automatic/smart cutting is not performed and this operation must be done beforehand
with the accuracy of a few samples.

- *ColorMap:integer* – returns the color map of the size specified in pixels; all other input arguments are ignored. Default: 1500.

- *WAV* - returns time warped output signal as a .wav file of 32 bit depth and sample rate of the reference signal.

- *Octave1/6* | *Octave1/12* - output of diffrogram images with corresponding frequency bands. Default: *Octave1/12*

- *WarpMargin:integer* - number of extra samples around the time window for more robust and accurate time warping. Default
value 5 is suitable for most cases but can be reduced for high frequency Sine signals if the time warping produces errors -
unreasonably low DF levels for some time frames. This is a development option.

Calling *diffrogram33(fref, fout, Tw, options)* without output argument returns diffrogram images and time warped output
signal (if warping was applied and WAV specified).

More details about diffrograms can be found in the article “Diffrogram: visualization of signal differences in audio research”
https://soundexpert.org/articles/-/blogs/visualization-of-distortion#part3
