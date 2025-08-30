% The code covered by BSD 2-Clause License.
% Copyright (c) 2025, Serge Smirnoff (soundexpert.org)
% Version 3.35; see the footer for ChangeLog.
% The reference implementation of df-metric (Matlab 2014a)
% Web: soundexpert.org/articles/-/blogs/visualization-of-distortion#part3
%
% The function compares two audio files which usually represent reference
% and output signals of some device under test.
%
% Usage examples:
% diffrogram33('ref.wav', 'out.wav', 100, 'Mono NoWarp')
% diffrogram33('ref.wav', 'out.wav', 400, 'Left SyncMargin:10')
% df = diffrogram33('ref.flac', 'out.wav', 400, 'Mono Octave1/6 WAV');
% diffrogram33(NaN, NaN, NaN, 'ColorMap:500')

%--------------------------------------------------------------------------

function df = diffrogram33(fref, fout, Tw, options)
disp('**********************************************')
disp('***            Diffrogram v3.35            ***')
disp('***            soundexpert.org             ***')
disp('**********************************************')

% Sample rate for time-warp processing
SRW = 574657; % Hz, the prime between close common multipliers of 44100 and
% 48000 (44100x13=573300 and 48000x12=576000). This makes equal errors of
% df computation with 44100 and 48000 REF signals.

options = lower(options);

% Return ColorMap only
if ~isempty(strfind(options, 'colormap'))
    si = textscan(options, '%s');
    tf = strncmp('colormap:', si{:}, 9);
    cc = char(si{:}(tf));
    NNmap = str2double(cc(10:end));
    if isnan(NNmap), NNmap = 1500; end % default value
    disp(['Color map output only, ' num2str(NNmap+1) 'x' num2str(NNmap) 'px'])
    colorM = cmap(NNmap);
    imwrite(colorM, 'diffrogram-color-map.png', 'bitdepth',16)
    df = NaN;
    return
end

% Frequency region of interest
% Vector of freq. bands for diffrogram images (indexes)
if ~isempty(strfind(options, 'octave1/6')) % 1/6 octave
    f01 = 440 .* (nthroot(2,6) .^ (-27:33)); % [19-19.9k]
    F0 = f01(1); F1 = f01(end);
    disp('1/6 octave bands diffrogram will be computed')
else % 1/12 octave
    f01 = 440 .* (nthroot(2,12) .^ (-54:66)); % [19-19.9k]
    F0 = f01(1); F1 = f01(end);
    disp('1/12 octave bands diffrogram will be computed')
end

[ref,Fsr] = audioread(fref);
[out,Fso] = audioread(fout);

% Define SyncMargin in milliseconds
if ~isempty(strfind(options, 'syncmargin:'))
    si = textscan(options, '%s');
    tf = strncmp('syncmargin:', si{:}, 11);
    cc = char(si{:}(tf));
    SyncMargin = str2double(cc(12:end));
else
    SyncMargin = 3000; % default value
end
% check its length
Lref = length(ref);
if round(SyncMargin*Fsr/1000) >= Lref
    SyncMs = Lref-1;
else
    SyncMs = round(SyncMargin*Fsr/1000);
end

% Cut OUT according to REF
if SyncMs ~= 0
    disp(['SyncMargin: ' num2str(SyncMs/Fsr*1000) ' ms'])
    [out,OSP] = CutAsRef(ref,Fsr,out,Fso,SyncMs);
    disp(['OSP: ' num2str(OSP)])
else
    disp('SyncMargin: 0 ms. Cutting OUT according to REF is NOT performed.')
    OverX = Fso / Fsr;
    Lut = length(out);
    Lef = Lref * OverX;
    OSP = round(Lef/(Lut-Lef));
    disp(['OSP: ' num2str(OSP)])
end

% Define the mode of operation: Left|Right|Mono
CHref = min(size(ref));
CHout = min(size(out));
if ~isempty(strfind(options, 'left'))
    if CHref == 1 && CHout == 2
        out = out(:,1);
    elseif CHref == 2 && CHout == 1
        ref = ref(:,1);
    elseif CHref == 1 && CHout == 1
        error('Either REF or OUT must have 2 channels in LEFT mode')
    elseif CHref == 2 && CHout == 2
        ref = ref(:,1);
        out = out(:,1);
    else
        error('More than 2 channels are not supported')
    end
    Channel = 'left';
    [ref,out] = NormByRef1(ref,out, Fsr,Fso, F0,F1);
elseif ~isempty(strfind(options, 'right'))
    if CHref == 1 && CHout == 2
        out = out(:,2);
    elseif CHref == 2 && CHout == 1
        ref = ref(:,2);
    elseif CHref == 1 && CHout == 1
        error('Either REF or OUT must have 2 channels in RIGHT mode')
    elseif CHref == 2 && CHout == 2
        ref = ref(:,2);
        out = out(:,2);
    else
        error('More than 2 channels are not supported')
    end
    Channel = 'right';
    [ref,out] = NormByRef1(ref,out, Fsr,Fso, F0,F1);
else % MONO mode
    if CHref == 1 && CHout == 2
        error('REF also must have 2 channels for 2ch OUT in MONO mode')
    elseif CHref == 2 && CHout == 1
        ref = (ref(:,1) + ref(:,2)) ./ 2;
        [ref,out] = NormByRef1(ref,out, Fsr,Fso, F0,F1);
    elseif CHref == 1 && CHout == 1
        [ref,out] = NormByRef1(ref,out, Fsr,Fso, F0,F1);
    elseif CHref == 2 && CHout == 2
        [ref(:,1),out(:,1)] = NormByRef1(ref(:,1),out(:,1), Fsr,Fso, F0,F1);
        [ref(:,2),out(:,2)] = NormByRef1(ref(:,2),out(:,2), Fsr,Fso, F0,F1);
        ref = (ref(:,1) + ref(:,2)) ./ 2;
        out = (out(:,1) + out(:,2)) ./ 2;
    else
        error('More than 2 channels are not supported')
    end
    Channel = 'mono';
end

disp(['FileRef: ' fref ', ' num2str(CHref) ' channel(s), ' num2str(Fsr) ' Hz'])
disp(['FileOut: ' fout ', ' num2str(CHout) ' channel(s), ' num2str(Fso) ' Hz'])
disp(['Channel mode: ' Channel])

% Define time window size for Df computing and time warping
if Tw < 1
    disp('The time window Tw can not be less than 1ms')
    return
end
Tws = round(Tw*Fsr/1000); % in samples of REF
if Tws >= Lref
    Tws = Lref;
    disp(['DF time window: the whole file (' num2str(Tws*1000/Fsr,'%12.2f') ' ms)'])
else
    disp(['DF time window: ' num2str(Tws*1000/Fsr) ' ms'])
end

% Freq. region of interest for the time window (sample indexes)
Twsmin = max(Tws, 16); % min. window for FFT = 2^3, #1/#4
Nq = ceil((Twsmin+1)/2);
F0s = round((Nq-1)/(Fsr/2)*F0) + 1;
F1s = round((Nq-1)/(Fsr/2)*F1) + 1;
if F0s==F1s, F1s = F1s + 1; end

% resulting vector of sample indexes
f01s = round((Nq-1)/(Fsr/2).*f01) + 1;

% Warp output audio (OUT) if required
if isempty(strfind(options, 'nowarp'))
    
    % Define WarpMargin in samples of REF
    if ~isempty(strfind(options, 'warpmargin:'))
        si = textscan(options, '%s');
        tf = strncmp('warpmargin:', si{:}, 11);
        cc = char(si{:}(tf));
        WarpMargin = str2double(cc(12:end));
        WarpMargin = abs(round(WarpMargin));
        if WarpMargin < 1, WarpMargin = 1; end
    else
        WarpMargin = 5; % default value
    end
    disp(['WarpMargin: ' num2str(WarpMargin) ' samples'])
    
    % Upsample OUT to SRW
    K = SRW / Fso;
    out = interpft(out,round(length(out)*K));
    
    % Lowpass upsampled OUT to Fcut
    Fcut = 24000; % Hz; Fcut = Fsr / 2; % Hz
    NN = ceil(SRW/2)*2; % even order of FIR filter
    b = fir1(NN, Fcut/(SRW/2));
    out = [out; zeros(NN/2,1)]; % pad OUT for group delay
    out = fftfilt(b,out);
    out = out(NN/2+1:end); % correct the padding
    
    % Lowpass REF to Fcut if required
    if Fsr/2 > Fcut
        MM = NN; % order of FIR filter
        b = fir1(MM, Fcut/(Fsr/2));
        ref = [ref; zeros(MM/2,1)]; % pad OUT for group delay
        ref = fftfilt(b,ref);
        ref = ref(MM/2+1:end); % correct the padding
    end
    
    % Searching local minimum of DF frame-by-frame
    Overlap = 0; % percent, 0 percent == no overlap
    Lout = length(out);
    SCL = Lout / Lref;
    
    % Slicing REF and OUT for parallel time warping
    OV = round(Tws/100*Overlap); if OV==Tws, OV = Tws-1; end
    NZ = round(Tws/100*Overlap/2);
    
    N = ceil((Lref-Tws)/(Tws-OV)) + 1;
    windx = zeros(N,4);
    refi = zeros(N,Tws);
    warpi = zeros(N,Tws);
    WWoutMax = ceil((Tws+2*WarpMargin)*SCL) + 1;
    outi = NaN(N,WWoutMax);
    
    Eref = 0;
    for ii = 1:N
        Bref = Eref - OV + 1;
        if Bref < 1, Bref = 1; end
        Eref = Bref + Tws - 1;
        if Eref > Lref
            Bref = Lref - Tws + 1;
            Eref = Lref;
        end
        Bout = round(SCL*(Bref-WarpMargin)); if Bout < 1, Bout = 1; end
        Eout = round(SCL*(Eref+WarpMargin)); if Eout > Lout, Eout = Lout; end
        
        windx(ii,1) = Bref; windx(ii,2) = Eref;
        windx(ii,3) = Bout; windx(ii,4) = Eout;
        
        refi(ii,:) = ref(Bref:Eref);
        outi(ii,1 : Eout-Bout+1) = out(Bout:Eout);
    end
    
    disp(['Warping: frame-wise (' num2str(Tws*1000/Fsr) ' ms)'])
    disp(['Number of frames: ' num2str(N)])
    disp('Time warping ...   (may take a long time)')
    
    % parallel time warping
    parfor ii=1:N
        samRef = refi(ii,:)';
        samOut = outi(ii,:)'; samOut = samOut(~isnan(samOut));
        
        [st,~] = fminsearch(@(st) DfWarp(samRef,samOut,st,WarpMargin,F0s,F1s,SCL), ...
            [0,0], optimset('TolX',1e-10, 'TolFun',1e-5, ...
            'Display','notify', 'MaxIter',1000, 'MaxFunEvals',2000));
        
        trw = 1:length(samOut);
        trw_u = trw(1)+st(2) : SCL+st(1) : trw(end)+st(2);
        samWarp = interp1(trw, samOut, trw_u', 'spline', 0);
        [~,samWarp] = CutByRef1(samRef,samWarp,WarpMargin);
        
        warpi(ii,:) = samWarp;
        
        [DFT,DFM,DFP,PP] = Df11(samRef,samWarp,F0s,F1s);
        disp(['Frame: ' num2str(ii) '/' num2str(N) ...
            '  DFwf: ' num2str(DFT,'%8.4f') ...
            '  DFmg: ' num2str(DFM,'%8.4f') ...
            '  DFph: ' num2str(DFP,'%8.4f') ...
            '  PP: ' num2str(PP,'%8.4f') ...
            '  SCL: ' num2str(SCL+st(1),'%10.6f') ...
            '  TAU: ' num2str(st(2),'%+8.4f') ...
            ])
    end
    
    % combine the slices into WARP (with overlapping if required)
    warp = zeros(size(ref));
    warp(windx(1,1):windx(1,2)) = warpi(1,:);
    for ii = 2:N
        warp(windx(ii,1)+NZ : windx(ii,2)) = warpi(ii,NZ+1:end);
    end
    [~,warp] = NormByRef1(ref,warp, Fsr,Fsr, F0,F1);
    
    % Output of warped audio file
    if ~isempty(strfind(options, 'wav'))
        wavname = [fout '(' num2str(floor(Fso/1000)) ')_' ...
            'warp_' Channel '_' num2str(round(Tws*1000/Fsr)) '.wav'];
        audiowrite(wavname, warp, Fsr, 'BitsPerSample',32);
    end
else % NoWarp case
    if Fso == Fsr
        disp('Warping: no')
        % warp = out;
        LagRef = 1 * Fsr; % aligne with 1s in the middle of REF
        if LagRef < Lref, LagRef = Lref; end
        [~,warp] = CutByRef1(ref,out,LagRef);
        clear out
        WarpMargin = 0; % for indication in diffrogram file name
    else
        disp('REF and OUT signals must have the same sample rate for NoWarp')
        df = NaN;
        return
    end
end

% Compute DF array (DFt,DFm,DFp,PP) and create diffrogram
disp('Computing DF values ...')
[df, dfmx] = dfMatrix(ref, warp, Tws, f01s);

disp('======================================== Waveform Df:')
[Max, Med, Min] = dispDfStat(df(:,1));
imdf = dfmx2img(dfmx(:,:,1), dfmx(:,:,4));
imgname = [ ...
    fref '(' num2str(floor(Fsr/1000)) ')_' ...
    fout '(' num2str(floor(Fso/1000)) ')_' ...
    Channel '_' num2str(Tws*1000/Fsr) ...
    '_Wf' num2str(Med,'%+8.2f') ...
    '[' num2str(Max,'%+8.2f') ...
    num2str(Min,'%+8.2f') ']_' ...
    'wM' num2str(WarpMargin) '_' ...
    'sM' num2str(SyncMargin) '_' ...
    'v335.png'];
imwrite(imdf, imgname, 'bitdepth',16)
disp('Waveform Diffrogram file:')
disp(imgname)

disp('======================================== Magnitude Df:')
[Max, Med, Min] = dispDfStat(df(:,2));
imdf = dfmx2img(dfmx(:,:,2), dfmx(:,:,4));
imgname = [ ...
    fref '(' num2str(floor(Fsr/1000)) ')_' ...
    fout '(' num2str(floor(Fso/1000)) ')_' ...
    Channel '_' num2str(Tws*1000/Fsr) ...
    '_Mg' num2str(Med,'%+8.2f') ...
    '[' num2str(Max,'%+8.2f') ...
    num2str(Min,'%+8.2f') ']_' ...
    'wM' num2str(WarpMargin) '_' ...
    'sM' num2str(SyncMargin) '_' ...
    'v335.png'];
imwrite(imdf, imgname, 'bitdepth',16)
disp('Magnitude Diffrogram file:')
disp(imgname)

disp('======================================== Phase Df:')
[Max, Med, Min] = dispDfStat(df(:,3));
imdf = dfmx2img(dfmx(:,:,3), dfmx(:,:,4));
imgname = [ ...
    fref '(' num2str(floor(Fsr/1000)) ')_' ...
    fout '(' num2str(floor(Fso/1000)) ')_' ...
    Channel '_' num2str(Tws*1000/Fsr) ...
    '_Ph' num2str(Med,'%+8.2f') ...
    '[' num2str(Max,'%+8.2f') ...
    num2str(Min,'%+8.2f') ']_' ...
    'wM' num2str(WarpMargin) '_' ...
    'sM' num2str(SyncMargin) '_' ...
    'v335.png'];
imwrite(imdf, imgname, 'bitdepth',16)
disp('Phase Diffrogram file:')
disp(imgname)

disp('Done.')
return

% ------------------------- sub-functions ---------------------------------

function Df = DfWarp(ref, out, st, WarpMargin, F0s,F1s, SCL)
if abs(st(1))>1, Df = 0; return; end % ~+-10% from SCL=SRW/Fsr
trw = 1:length(out);
trw_u = trw(1)+st(2) : SCL+st(1) : trw(end)+st(2);
warp = interp1(trw, out, trw_u', 'spline', 0);
[~,warp] = CutByRef1(ref,warp,WarpMargin);
n = max(length(ref),16); % min. window for FFT = 2^3, #2/#4
fref = fft(ref, n);
fout = fft(warp, n);
r = corrcoef(fref(F0s:F1s),fout(F0s:F1s)); r = real(r);
if ~isnan(r(1,1)) && ~isnan(r(1,2)) && ~isnan(r(2,1)) && ~isnan(r(2,2))
    %both "ref" and "out" are NOT constant
    Df = 10*log10(1-r(1,2));
elseif isnan(r(1,1)) && isnan(r(1,2)) && isnan(r(2,1)) && isnan(r(2,2))
    %both "ref" and "out" are constant
    Df = -Inf;
else
    %only "ref" or only "out" is constant
    Df = 0;
end
if isinf(Df), Df = -9999; end
return


function [ref,out] = CutByRef1(ref, out, Lag)
Dlen = length(ref)-length(out);
DL = abs(Dlen);
HDB = floor(DL/2);
HDE = DL - HDB;

if Dlen < 0
    ref = [zeros(HDB,1); ref; zeros(HDE,1)];
elseif 	Dlen > 0
    out = [zeros(HDB,1); out; zeros(HDE,1)];
end

[~,Maxind] = max(xcorr(ref,out,Lag));

if Maxind < (Lag+1)
    Shift = (Lag+1) - Maxind;
    out = [out(Shift+1:end,:);zeros(Shift,1)];
elseif Maxind > (Lag+1)
    Shift = Maxind - (Lag+1);
    out = [zeros(Shift,1);out(1:end-Shift,:)];
end

if Dlen < 0
    ref = ref(1+HDB:end-HDE,:);
    out = out(1+HDB:end-HDE,:);
end
return


function [DFt,DFm,DFp,PP] = Df11(ref,out,F0s,F1s)
Lref = length(ref);
n = max(Lref,16); % min. window for FFT = 2^3, #3/#4
fref = fft(ref, n);
fout = fft(out, n);
r = corrcoef(fref(F0s:F1s),fout(F0s:F1s)); r = real(r);
if ~isnan(r(1,1)) && ~isnan(r(1,2)) && ~isnan(r(2,1)) && ~isnan(r(2,2))
    % both "ref" and "out" are NOT constant
    %
    % Waveform
    DFt = 10*log10(1-r(1,2));
    % Magnitude
    rM = corrcoef(abs(fref(F0s:F1s)),abs(fout(F0s:F1s)));
    DFm = 10*log10(1-rM(1,2));
    % Phase
    dp = abs(angle(fref(F0s:F1s))-angle(fout(F0s:F1s)));
    dp(dp>pi) = 2*pi - dp(dp>pi);
    rP = mean(dp.^2) / (pi^2/3);
    DFp = 10*log10(rP);
    % PSD per band (0dB, Square)
    PP = 2*sum(abs(fout(F0s:F1s)./sqrt(n*Lref)).^2);
    PP = 10*log10(PP);
elseif isnan(r(1,1)) && isnan(r(1,2)) && isnan(r(2,1)) && isnan(r(2,2))
    %both "ref" and "out" are constant
    %
    DFt = -Inf; DFm = -Inf; DFp = -Inf;
    % PSD per band (0dB, Square)
    PP = 2*sum(abs(fout(F0s:F1s)./sqrt(n*Lref)).^2);
    PP = 10*log10(PP);
else
    %only "ref" or only "out" is constant
    %
    DFt = 0; DFm = 0; DFp = 0;
    % PSD per band (0dB, Square)
    PP = 2*sum(abs(fout(F0s:F1s)./sqrt(n*Lref)).^2);
    PP = 10*log10(PP);
end
return


function [df,dfmx] = dfMatrix(ref, out, Wdiff, f01s)

Lref = length(ref);
N = floor(Lref/Wdiff);
Lfs = length(f01s);
if rem(Lref,Wdiff) == 0
    df = zeros(N,4);
    dfmx = zeros(N,Lfs-1,4);
else
    df = zeros(N+1,4);
    dfmx = zeros(N+1,Lfs-1,4);
end

E = 0;
for ii = 1:N
    B = E + 1;
    E = B + Wdiff - 1;
    samRef = ref(B:E);
    samOut = out(B:E);
    
    [df(ii,1),df(ii,2),df(ii,3),df(ii,4)] = Df11(samRef,samOut,f01s(1),f01s(end));
    
    Lsam = length(samRef);
    n = max(Lsam,16); % min. window for FFT = 2^3, #4/#4
    fref = fft(samRef, n);
    fout = fft(samOut, n);
    for jj = 1:Lfs-1
        B0 = f01s(jj); B1 = f01s(jj+1);
        if B1-B0<2, B1 = B0+2; end
        bref = fref(B0:B1); bout = fout(B0:B1);
        r = corrcoef(bref,bout); r = real(r);
        if ~isnan(r(1,1)) && ~isnan(r(1,2)) && ~isnan(r(2,1)) && ~isnan(r(2,2))
            %both "ref" and "out" are NOT constant
            %
            % Waveform
            dfmx(ii,jj,1) = 10*log10(1-r(1,2));
            % Magnitude
            rM = corrcoef(abs(bref),abs(bout));
            dfmx(ii,jj,2) = 10*log10(1-rM(1,2));
            % Phase
            dp = abs(angle(bref)-angle(bout));
            dp(dp>pi) = 2*pi - dp(dp>pi);
            rP = mean(dp.^2) / (pi^2/3);
            dfmx(ii,jj,3) = 10*log10(rP);
            % PSD per band (0dB, Square)
            Pband = 2*sum(abs(bout./sqrt(n*Lsam)).^2);
            dfmx(ii,jj,4) = 10*log10(Pband);
        elseif isnan(r(1,1)) && isnan(r(1,2)) && isnan(r(2,1)) && isnan(r(2,2))
            %both "ref" and "out" are constant
            dfmx(ii,jj,1) = -Inf; dfmx(ii,jj,2) = -Inf; dfmx(ii,jj,3) = -Inf;
            % PSD per band (0dB, Square)
            Pband = 2*sum(abs(bout./sqrt(n*Lsam)).^2);
            dfmx(ii,jj,4) = 10*log10(Pband);
        else
            %only "ref" or only "out" is constant
            dfmx(ii,jj,1) = 0; dfmx(ii,jj,2) = 0; dfmx(ii,jj,3) = 0;
            % PSD per band (0dB, Square)
            Pband = 2*sum(abs(bout./sqrt(n*Lsam)).^2);
            dfmx(ii,jj,4) = 10*log10(Pband);
        end
    end
end

% last time window
if rem(Lref,Wdiff) ~= 0
    B = Lref - Wdiff + 1;
    E = Lref;
    samRef = ref(B:E);
    samOut = out(B:E);
    
    [df(end,1),df(end,2),df(end,3),df(ii,4)] = Df11(samRef,samOut,f01s(1),f01s(end));
    
    fref = fft(samRef, n);
    fout = fft(samOut, n);
    for jj = 1:Lfs-1
        B0 = f01s(jj); B1 = f01s(jj+1);
        if B1-B0<2, B1 = B0+2; end
        bref = fref(B0:B1); bout = fout(B0:B1);
        r = corrcoef(bref,bout); r = real(r);
        if ~isnan(r(1,1)) && ~isnan(r(1,2)) && ~isnan(r(2,1)) && ~isnan(r(2,2))
            %both "ref" and "out" are NOT constant
            %
            % Waveform
            dfmx(end,jj,1) = 10*log10(1-r(1,2));
            % Magnitude
            rM = corrcoef(abs(bref),abs(bout));
            dfmx(end,jj,2) = 10*log10(1-rM(1,2));
            % Phase
            dp = abs(angle(bref)-angle(bout));
            dp(dp>pi) = 2*pi - dp(dp>pi);
            rP = mean(dp.^2) / (pi^2/3);
            dfmx(ii,jj,3) = 10*log10(rP);
            % PSD per band (0dB, Square)
            Pband = 2*sum(abs(bout./sqrt(n*Lsam)).^2);
            dfmx(end,jj,4) = 10*log10(Pband);
        elseif isnan(r(1,1)) && isnan(r(1,2)) && isnan(r(2,1)) && isnan(r(2,2))
            %both "ref" and "out" are constant
            %
            dfmx(end,jj,1) = -Inf; dfmx(end,jj,2) = -Inf; dfmx(end,jj,3) = -Inf;
            % PSD per band (0dB, Square)
            Pband = 2*sum(abs(bout./sqrt(n*Lsam)).^2);
            dfmx(end,jj,4) = 10*log10(Pband);
        else
            %only "ref" or only "out" is constant
            %
            dfmx(end,jj,1) = 0; dfmx(end,jj,2) = 0; dfmx(end,jj,3) = 0;
            % PSD per band (0dB, Square)
            Pband = 2*sum(abs(bout./sqrt(n*Lsam)).^2);
            dfmx(end,jj,4) = 10*log10(Pband);
        end
    end
end
return


function imdf = dfmx2img(df, sp)
% Create image (png16) of diffrogram
MinDf = -150; % dB, violet
MinPw = -150; % dB, black
Nmap = 1500;

df(df<MinDf & ~isinf(df)) = MinDf;
df(df>0) = 0;
df = round(Nmap + df .* ((Nmap-1)/Nmap*10)) + 1;
df(isinf(df)) = 1;

sp(sp<MinPw) = MinPw;
sp(sp>0) = 0;
sp = round(1 - sp .* ((Nmap-1)/Nmap*10));

colorM = cmap(Nmap);
sizeDf = size(df);

imdf = zeros(sizeDf(2),sizeDf(1),3);
for ii = 1:sizeDf(1)
    for jj = 1:sizeDf(2)
        imdf(sizeDf(2)-jj+1,ii,:) = colorM(sp(ii,jj),df(ii,jj),:);
    end
end
return


function colorM = cmap(Nmap)
% Build ColorMap with Grey scale for -Inf [dB]
m = [0.718, 0.000, 0.718];
b = [0.316, 0.316, 0.991];
c = [0.000, 0.559, 0.559];
g = [0.000, 0.592, 0.000];
y = [0.527, 0.527, 0.000];
r = [0.847, 0.057, 0.057];

gamma = 2;
Nseg = round(Nmap/5);

lin = linspace(0,1,Nseg);
mb = zeros(Nseg,3);
mb(:,1) = ((1-lin).*m(1)^gamma + lin.*b(1)^gamma).^(1/gamma);
mb(:,2) = ((1-lin).*m(2)^gamma + lin.*b(2)^gamma).^(1/gamma);
mb(:,3) = ((1-lin).*m(3)^gamma + lin.*b(3)^gamma).^(1/gamma);

lin = linspace(0,1,Nseg+1);
bc = zeros(Nseg+1,3);
bc(:,1) = ((1-lin).*b(1)^gamma + lin.*c(1)^gamma).^(1/gamma);
bc(:,2) = ((1-lin).*b(2)^gamma + lin.*c(2)^gamma).^(1/gamma);
bc(:,3) = ((1-lin).*b(3)^gamma + lin.*c(3)^gamma).^(1/gamma);

cg = zeros(Nseg+1,3);
cg(:,1) = ((1-lin).*c(1)^gamma + lin.*g(1)^gamma).^(1/gamma);
cg(:,2) = ((1-lin).*c(2)^gamma + lin.*g(2)^gamma).^(1/gamma);
cg(:,3) = ((1-lin).*c(3)^gamma + lin.*g(3)^gamma).^(1/gamma);

gy = zeros(Nseg+1,3);
gy(:,1) = ((1-lin).*g(1)^gamma + lin.*y(1)^gamma).^(1/gamma);
gy(:,2) = ((1-lin).*g(2)^gamma + lin.*y(2)^gamma).^(1/gamma);
gy(:,3) = ((1-lin).*g(3)^gamma + lin.*y(3)^gamma).^(1/gamma);

yr = zeros(Nseg+1,3);
yr(:,1) = ((1-lin).*y(1)^gamma + lin.*r(1)^gamma).^(1/gamma);
yr(:,2) = ((1-lin).*y(2)^gamma + lin.*r(2)^gamma).^(1/gamma);
yr(:,3) = ((1-lin).*y(3)^gamma + lin.*r(3)^gamma).^(1/gamma);

colorV = [mb; bc(2:end,:); cg(2:end,:); gy(2:end,:); yr(2:end,:)];

colorR = smooth(colorV(:,1),Nseg);
colorG = smooth(colorV(:,2),Nseg);
colorB = smooth(colorV(:,3),Nseg);

colorV = [colorR colorG colorB];

Klum = mean(colorV(:));
Ndark = round(Nmap*Klum);
Nlight = Nmap - Ndark + 1;

colorM = zeros(Nmap, Nmap+1, 3);
colorM(1:Nmap,1,:) = interp1([1;Nmap],[1 1 1; 0 0 0],1:Nmap);
for ii = 1:Nmap
    colorM(1:Nlight,ii+1,:) = interp1([1;Nlight],[1 1 1; colorV(ii,:)],1:Nlight);
    colorM(Nlight:end,ii+1,:) = interp1([1;Ndark],[colorV(ii,:); 0 0 0],1:Ndark);
end
return


function [ref,out] = NormByRef1(ref,out, Fsr, Fso, F0,F1)

ref = detrend(ref,'constant');
out = detrend(out,'constant');

fref = fft(ref);
fout = fft(out);

% Energy of REF in the freq. region of interest
Lref = length(ref);
Nq = ceil((Lref+1)/2);
F0s = round((Nq-1)/(Fsr/2)*F0) + 1;
F1s = round((Nq-1)/(Fsr/2)*F1) + 1;
bref = fref(F0s:F1s) ./ Lref;
Px = sqrt(mean(abs(bref).^2));
% Px = sqrt(mean(ref.^2));

% Energy of OUT in the freq. region of interest
Lout = length(out);
Nq = ceil((Lout+1)/2);
F0s = round((Nq-1)/(Fso/2)*F0) + 1;
F1s = round((Nq-1)/(Fso/2)*F1) + 1;
bout = fout(F0s:F1s) ./ Lout;
Py = sqrt(mean(abs(bout).^2));
% Py = sqrt(mean(out.^2));

Kp = Px / Py;
out = out .* Kp;
return


function [Max, Med, Min] = dispDfStat(df)
Ndf = length(df);
if Ndf == 1
    disp(['Resulting Df statistics (' num2str(Ndf) ' value):'])
    Max = df;
    Med = df;
    Min = df;
    disp(['Df = ' num2str(df,'%+8.4f') ' dB'])
elseif Ndf == 2
    disp(['Resulting Df statistics (' num2str(Ndf) ' values):'])
    Max = max(df);
    Med = median(df);
    Min = min(df);
    disp(['Df.first = ' num2str(df(1),'%+8.4f') ' dB'])
    disp(['Df.last = ' num2str(df(2),'%+8.4f') ' dB'])
elseif Ndf == 3
    disp(['Resulting Df statistics (' num2str(Ndf) ' values):'])
    Max = max(df);
    Med = median(df);
    Min = min(df);
    disp(['Df.first = ' num2str(df(1),'%+8.4f') ' dB'])
    disp(['Df = ' num2str(df(2),'%+8.4f') ' dB'])
    disp(['Df.last = ' num2str(df(3),'%+8.4f') ' dB'])
elseif Ndf == 4
    disp(['Resulting Df statistics (' num2str(Ndf) ' values):'])
    Max = max(df(2:end-1));
    Med = median(df(2:end-1));
    Min = min(df(2:end-1));
    disp(['Df.first = ' num2str(df(1),'%+8.4f') ' dB'])
    disp(['Df1 = ' num2str(df(2),'%+8.4f') ' dB'])
    disp(['Df2 = ' num2str(df(3),'%+8.4f') ' dB'])
    disp(['Df.last = ' num2str(df(4),'%+8.4f') ' dB'])
else
    disp(['Resulting Df statistics (' num2str(Ndf) ' values):'])
    % statistics without the first and the last Df values
    Max = max(df(2:end-1));
    P75 = prctile(df(2:end-1),75); if isnan(P75), P75 = -Inf; end
    Med = median(df(2:end-1));
    P25 = prctile(df(2:end-1),25); if isnan(P25), P25 = -Inf; end
    Min = min(df(2:end-1));
    disp(['Df.first = ' num2str(df(1),'%+8.4f') ' dB'])
    disp(['Df.last = ' num2str(df(end),'%+8.4f') ' dB'])
    disp(['For remaining ' num2str(Ndf-2) ' values:'])
    disp(['Df.max = ' num2str(Max,'%+8.4f') ' dB'])
    disp(['Df.p75 = ' num2str(P75,'%+8.4f') ' dB'])
    disp(['Df.median = ' num2str(Med,'%+8.4f') ' dB'])
    disp(['Df.p25 = ' num2str(P25,'%+8.4f') ' dB'])
    disp(['Df.min = ' num2str(Min,'%+8.4f') ' dB'])
end
return

function [out,OSP] = CutAsRef(ref,Fsr,out,Fso,W)
% W = 40000; % samples of REF, correlation window for cutting OUT as REF
M = 0; % samples of OUT, +extra margin on both sides of OUT

OverX = Fso / Fsr;
if OverX > 1, ref = interpft(ref,round(length(ref)*OverX)); end
if OverX < 1, [P,Q] = rat(Fso/Fsr); ref = resample(ref,P,Q); end

if min(size(out)) == 2
    Ltest = round(W * OverX);
    if min(size(ref)) == 2, testr = ref(1:Ltest,1) + ref(1:Ltest,2); end
    if min(size(ref)) == 1, testr = ref(1:Ltest); end
    
    testo = out(1:Ltest,1) + out(1:Ltest,2);
    [c,lags] = xcorr(testr,testo);
    [~,Maxind] = max(c);
    Lag = lags(Maxind);
    Lag = Lag + M;
    if Lag < 0
        out = out(1-Lag:end,:);
    else
        out = [zeros(Lag,2); out];
    end
    
    if min(size(ref)) == 2, testr = ref(end-Ltest+1:end,1) + ref(end-Ltest+1:end,2); end
    if min(size(ref)) == 1, testr = ref(end-Ltest+1:end); end
    testo = out(end-Ltest+1:end,1) + out(end-Ltest+1:end,2);
    [c,lags] = xcorr(testr,testo);
    [~,Maxind] = max(c);
    Lag = lags(Maxind);
    Lag = Lag - M;
    if Lag < 0
        out = [out;zeros(-Lag,2)];
    else
        out = out(1:end-Lag,:);
    end
    
    Lref = length(ref); Lout = length(out);
    OSP = round(Lref/(Lout-Lref));
end

if min(size(out)) == 1
    Ltest = round(W * OverX);
    if min(size(ref)) == 2, testr = ref(1:Ltest,1) + ref(1:Ltest,2); end
    if min(size(ref)) == 1, testr = ref(1:Ltest); end
    
    testo = out(1:Ltest);
    [c,lags] = xcorr(testr,testo);
    [~,Maxind] = max(c);
    Lag = lags(Maxind);
    Lag = Lag + M;
    if Lag < 0
        out = out(1-Lag:end);
    else
        out = [zeros(Lag,1); out];
    end
    
    if min(size(ref)) == 2, testr = ref(end-Ltest+1:end,1) + ref(end-Ltest+1:end,2); end
    if min(size(ref)) == 1, testr = ref(end-Ltest+1:end); end
    testo = out(end-Ltest+1:end);
    [c,lags] = xcorr(testr,testo);
    [~,Maxind] = max(c);
    Lag = lags(Maxind);
    Lag = Lag - M;
    if Lag < 0
        out = [out;zeros(-Lag,1)];
    else
        out = out(1:end-Lag);
    end
    
    Lref = length(ref); Lout = length(out);
    OSP = round(Lref/(Lout-Lref));
end

if min(size(out)) > 2
    disp('More than two channels are not supported')
end

return

% ---------------------------- Changelog ----------------------------------
%
% V3.36
% - BugFix: SyncMargin now correctly handles the OUT files of lower
% sampling rates.
% V3.35
% - Custom filtering of OUT with mg2020.mat was removed.
% - now names of output images have REF file first, then OUT. WarpMargin
% and SyncMargin are indicated as well.
% - Optimization of oversampling rate (SRW) for accuracy improvement
% V3.34
% - Phase Df computation changed, to be more consistent with Wf and Mg
% scales
% - Low-pass filtering of REF and OUT at 24000 Hz before time warping
% - Custom filtering of OUT with mg2020.mat before warping
% - OSP (One Sample Period) indication
% - SyncMargin is in milliseconds now
% - The Time diffrogram is now called The Waveform diffrogram
% V3.33
% - Correct normalization of OUT, according to the energy in the freq.
%   region of interest only
% - Correct computing of 1/12 and 1/6 bands in case of short freq. regions
%   of interest
% - names of output images changed to be more convenient
% V3.32
% - RMS power as a 4th vector in the output array
% - fall back of min FFT to "defined by the length of a signal" (affects
%   images only)
% V3.31
% - bug fix in case of mono OUT signal
% - min FFT = 16384 (2^14); it was 8192
% V3.3
% - computing of Magnitude and Phase DF levels added
% - full-featured diffrograms with 1/6 or 1/12 octave bands
% - time warping with overlapping
% - DF computing for freq. region of interest: 20Hz - 20kHz
% - the error of DF computation with white noise is less than 0.1dB
%   at DF = -100dB; the error of DF computation with PSN is less than 0.1dB
%   at DF = -120dB.
% - faster processing due to efficient lowpass filtering and parallel time
%   warping
% V2.4
% - removed normalisation of input streams (redundant operation)
% - redisigned initial resampling of OUT signal; now parameters of
%   resampling are chosen autimaticly to get equal accuracy at various
%   samplerates with pseudo-white noise down to -100 dB
% - automatic alignment of REF & OUT in case of "NoWarp" and diff. lengths
% - no output of time-warped WAV files by-default
% V2.3
% - default warping frame size = Df window size
% - display statistics at the end
% - less output of debug info
% - optimization of spectrogram computing (shorter fft window)
% - new "NoUpsample" option
% - option STEREO removed
% V2.2
% - bugfix lower case for "ColorMap" option
% - direct colormap output without df and sp vectors
% - added -Inf column (greyscale) to color map
% - first public release


