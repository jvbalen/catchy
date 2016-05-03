function psy_features(in_folder, out_folder, mirtoolbox_path)
% PSY_FEATURES(IN_FOLDER, OUR_FOLDER, MIRTOOLBOX_PATH)
% Computes and writes loudness, sharpness and roughness features
%   for all wav and mp3 files in folder IN_FOLDER.
% Features are written to CSV files, one per song, one folder per feature.
% 
% Arguments:
%   IN_FOLDER: where to find audio files
%   OUT_FOLDER: where feature data should be written to.
%   MIRTOOLBOX_PATH: the path to the MIRTOOLBOX.
% 
% Requires MIR Toolbox by Olivier Lartillot, and includes a part of the
%   MA Toolbox by Elias Pampalk.
%   
% Requires a folder 'loudness/', 'sharpness/', and 'roughness/' in
%   OUT_FOLDER.
% 
% Distributed under the GNU General Public License.

    sr = 22050;

    if nargin < 3
        mirtoolbox_path = '/Users/Jan/Documents/Work/Matlab/mirtoolbox/MIRToolbox';
    end
    addpath(mirtoolbox_path)

    % Analyze all tracks in in_folder
    audio_files = [dir([in_folder '*.wav']); dir([in_folder '*.mp3'])]';
    for file = audio_files

        disp(['Extracting features for ' file.name '...'])
        in_file = [in_folder file.name];
        filename_split = strsplit(file.name, '.');
        track_id = filename_split{end-1};

        % Read audio and compute roughness using MIR Toolbox
        audio = miraudio(in_file, 'Sampling', sr);
        roughnessdata = mirroughness(audio);
        roughness = mirgetdata(roughnessdata)';
        
        nframes = length(roughness);
        t_roughness = (0.5:nframes)'*0.025;

        % Compute loudness and sharpness
        x = mirgetdata(audio);
        [t_bark, loudness, sharpness] = barkfeatures(x, sr);

        % Write to CSV
        loudness_file = [out_folder 'loudness/' track_id '.csv'];
        sharpness_file = [out_folder 'sharpness/' track_id '.csv'];
        roughness_file = [out_folder 'roughness/' track_id '.csv'];

        csvwrite(loudness_file, [t_bark loudness]);
        csvwrite(sharpness_file, [t_bark sharpness]);
        csvwrite(roughness_file, [t_roughness roughness]);
    end

end

function [t_bark, loudness, sharpness] = barkfeatures(x, sr)
% [t_bark, loudness, sharpness] = BARKFEATURES(X, SR)
% 
%   Computes loudness and sharpness for audio data X with samplerate SR.
%   LOUDNESS contains the total loudness
%   SHARPNESS contains the sharpness as described by Peeters
%     in the CUIDADO project
%   T_BARK contains the time stamps for the rows of LOUDNESS and SHARPNESS.


    % Compute Loudness
    disp('Computing loudness...')
    
    p = struct;
    p.fs = sr;
    [n N p_out] = ma_sone(x, p);
    
    hop = p_out.hopsize;
    [nbands nframes] = size(n);
    t_bark = (0.5:nframes)'*hop/sr;

    loudness = N;
    
    % Compute Sharpness
    disp('Computing sharpness...')
    
    z1 = 1:15;
    z2 = 16:nbands;
    Z = diag([z1 z2]);
    g1 = ones(size(z1)); 
    g2 = 0.066*exp(0.171*z2); 
    G = diag([g1 g2]);
    
    sharpness = 0.22*sum(n'*G*Z,2)./N;
    sharpness(isnan(sharpness))=0;

end

function [sone, Ntot, p] = ma_sone(wav,p)
% Code from the MA Toolbox by Elias Pampalk to compute bark band loudness.
% See: http://www.pampalk.at/ma/index.html
% 
% Distributed under the GNU General Public License.
%
% Compute sonogram for a pcm signal.
%    [sone, Ntot, p] = ma_sone(wav,p)
%
% USAGE
%   ma_sone;                % uses test-sound and creates figures
%   sone = ma_sone(wav,p);  % normal usage
%   [sone, Ntot, p] = ma_sone(wav,p);
%
% INPUT
%   wav (vector) obtained from wavread (use mono input! 11kHz recommended)
%   p (struct) parameters e.g. 
%    p.fs          = 11025;       % sampling frequency of given wav (unit: Hz)
%    * p.do_visu     = 0;         % create some figures
%    * p.do_sone     = 1;         % compute sone (otherwise dB)
%    * p.do_spread   = 1;         % apply spectral masking
%    * p.outerear = 'terhardt';   % outer ear model {'terhardt' | 'none'}
%    * p.fft_size = 256;          % window size (unit: samples) 256 are ~23ms @ 11kHz 
%    * p.hopsize  = 128;          % fft window hopsize (unit: samples)
%    * p.bark_type   = 'table';   % type of bark scale to use (either:
%                                 % 'table' lookup (=default) with max 24 bands 
%                                 % (for 44kHz), or vector [min_freq max_freq num_bands]
%    * p.dB_max      = 96;        % max dB of input wav (for 16 bit input 96dB is SPL)
%    all fields marked with * are optional (defaults are defined)
%                                 
% OUTPUT
%   sone (matrix) rows are bark bands, columns are time intervalls, 
%                 values are sone (or dB) 
%   Ntot (vector) total loudness (stevens method, see "signal sound 
%                 and sensation" p73, hartmann)

% elias 13.6.2004

    if ~nargin, % run test
        disp('testing: ma_sone')        

        p.type      = 'sone';
        p.outerear  = 'terhardt'; % {'terhardt' | 'none'}
        p.fft_size  = 512;        
        p.hopsize   = 256;        
        p.do_visu   = 1;
        p.do_sone   = 1;
        p.do_spread = 1;
        p.fs        = 22050;
        p.bark_type = 'table'; % {'table' | e.g. [20 20000 50]}
        p.dB_max    = 96;

        wav = ma_test_create_wav(p.fs);
        sone = ma_sone(wav,p);

        %sound(wav,p.fs);

        sone = 'done'; % dont flood command window with numbers
        return
    end 

    % defaults
    if ~isfield(p,'type'),      p.type = 'sone';         end
    if ~isfield(p,'outerear'),  p.outerear = 'terhardt'; end
    if ~isfield(p,'bark_type'), p.bark_type = 'table';   end
    if ~isfield(p,'dB_max'),    p.dB_max = 96;           end
    if ~isfield(p,'do_visu'),   p.do_visu = 0;           end
    if ~isfield(p,'do_sone'),   p.do_sone = 1;           end
    if ~isfield(p,'do_spread'), p.do_spread = 1;         end
    if ~isfield(p,'fs'),        error('sampling frequency (p.fs) not specified'); end
    if ~isfield(p,'fft_size'),  p.fft_size = p.fs/11025*256; end
    if ~isfield(p,'hopsize'),   p.hopsize = p.fs/11025*128;  end

    % frequency of fft bins
    c.fft_freq = (0:p.fft_size/2)/p.fft_size*2*p.fs/2;

    if strcmp(p.bark_type,'table'),
        % zwicker & fastl: psychoacoustics 1999, page 159
        c.bark_upper = [10 20 30 40 51 63 77 92 108 127 148 172 200 232 270 315 370 440 530 640 770 950 1200 1550]*10; % Hz
        c.bark_center = [5 15 25 35 45 57 70 84 100 117 137 160 185 215 250 290 340 400 480 580 700 850 1050 1350]*10; % Hz

        % ignore critical bands outside of p.fs range
        cb = min(min([find(c.bark_upper>p.fs/2),length(c.bark_upper)]),length(c.bark_upper));
        c.bark_center = c.bark_center(1:cb);
    else
        cb = p.bark_type(3);
        if ~isnumeric(cb) | ceil(cb)~=cb | cb<2,
            error('p.bark_type should be {''table''| 2..50})');
        end
        tmp.f = p.bark_type(1):min(p.bark_type(2),p.fs/2);
        tmp.bark = 13*atan(0.76*tmp.f/1000) + 3.5*atan((tmp.f/7500).^2);
        tmp.f_idx_upper = (1:cb)*0;
        tmp.b_idx_upper = linspace(1,max(tmp.bark),cb); 
        tmp.f_idx_center = (1:cb)*0;
        tmp.b_idx_center = tmp.b_idx_upper-diff(tmp.b_idx_upper(1:2))/2;  
        for i=1:cb,
            [dummy tmp.f_idx_upper(i)] = min(abs(tmp.bark - tmp.b_idx_upper(i)));
            [dummy tmp.f_idx_center(i)] = min(abs(tmp.bark - tmp.b_idx_center(i)));
        end
        c.bark_upper = tmp.f(tmp.f_idx_upper);
        c.bark_center = tmp.f(tmp.f_idx_center);
    end

    % spreading function: schroeder et al., 1979, JASA, optimizing digital speech coders by exploiting masking properties of the human ear
    for i = 1:cb, 
        c.spread(i,:) = 10.^((15.81+7.5*((i-(1:cb))+0.474)-17.5*(1+((i-(1:cb))+0.474).^2).^0.5)/10);
    end

    switch p.outerear,
        case 'terhardt', % terhardt 1979 (calculating virtual pitch, hearing research #1, pp 155-182)
            c.W_Adb(1) = 0;
            c.W_Adb(2:length(c.fft_freq)) = ...
                + 10.^((-3.64*(c.fft_freq(2:end)/1000).^-0.8 ...
                + 6.5 * exp(-0.6 * (c.fft_freq(2:end)/1000 - 3.3).^2) ...
                - 0.001*(c.fft_freq(2:end)/1000).^4)/20);
            c.W_Adb = c.W_Adb.^2;
        case 'modified_terhardt', % less emph around 4Hz, more emphasis on low freqs
            c.W_Adb(1) = 0;
            c.W_Adb(2:length(c.fft_freq)) = ...
                + 10.^((.6*-3.64*(c.fft_freq(2:end)/1000).^-0.8 ...
                + 0.5 * exp(-0.6 * (c.fft_freq(2:end)/1000 - 3.3).^2) ...
                - 0.001*(c.fft_freq(2:end)/1000).^4)/20);
            c.W_Adb = c.W_Adb.^2;
        case 'none', % all weighted equally
            c.W_Adb(1:length(c.fft_freq)) = 1; 
        otherwise error(['unknown outer ear model: p.outerear = ',p.outerear]);
    end

    if p.do_visu, % create figs of psychoacoustic models        

        tmp.cbW_T = -3.64*(c.bark_center/1000).^-0.8 ...
            + 6.5 * exp(-0.6 * (c.bark_center/1000 - 3.3).^2) ...
            - 0.001*(c.bark_center/1000).^4; 

        tmp.W_Adb_T(1) = 0;
        tmp.W_Adb_T(2:length(c.fft_freq)) = 10.^((-3.64*(c.fft_freq(2:end)/1000).^-0.8 + 6.5 * exp(-0.6 * (c.fft_freq(2:end)/1000 - 3.3).^2) - 0.001*(c.fft_freq(2:end)/1000).^4)/20);
        tmp.W_Adb_T = tmp.W_Adb_T.^2;

    %     tmp.cbW_mT = .6*-3.64*(c.bark_center/1000).^-0.8 ...
    %         + 6.5 * exp(-0.6 * (c.bark_center/1000 - 3.3).^2) ...
    %         - 0.001*(c.bark_center/1000).^4;     
    %     
    %     tmp.W_Adb_mT(1) = 0;
    %     tmp.W_Adb_mT(2:length(c.fft_freq)) = 10.^((.6*-3.64*(c.fft_freq(2:end)/1000).^-0.8 + 0.5 * exp(-0.6 * (c.fft_freq(2:end)/1000 - 3.3).^2) - 0.001*(c.fft_freq(2:end)/1000).^4)/20);
    %     tmp.W_Adb_mT = tmp.W_Adb_mT.^2;    

        figure % psychoacoustic model

        subplot(2,1,1); % outer ear weighting function and width of bark bands 
        set(gca,'fontsize',8);
        semilogx(c.fft_freq(2:end),10*log10(tmp.W_Adb_T(2:end)),'r'); hold on
    %    semilogx(c.fft_freq(2:end),10*log10(tmp.W_Adb_mT(2:end)),'b'); hold on
        legend({'Terhardt'},2)
    %    legend({'Terhardt','mod. Terhardt'},2)
        set(gca,'ylim',[-50 10],'xlim',[30 16e3])
        set(gca,'xtick',[50,100,200,400,800,1600,3200,6400,12800],'XMinorTick','off')
        plot(cb,c.W_Adb,'.k');
        for i=1:cb,
            plot([c.bark_center(i) c.bark_center(i)],[-50 tmp.cbW_T(i)],':k')
        end
        ylabel('Response [dB]'); title('Outer Ear'); xlabel('Frequency [Hz]')

        subplot(2,2,3); % bark-scale
        z = 13*atan(0.76*c.fft_freq/1000)+3.5*atan(c.fft_freq/7.5/1000).^2;
        cbz = 13*atan(0.76*(c.bark_center(1:cb))/1000)+3.5*atan((c.bark_center(1:cb))/7.5/1000).^2;
        set(gca,'fontsize',8)
        plot(c.fft_freq,z); hold on
        plot((c.bark_center(1:cb)),cbz,'.r')
        xlabel('Hz'); ylabel('Bark'); title('Bark Scale')

        subplot(2,2,4); % spreading function
        i = 10; set(gca,'fontsize',8)
        plot(((15.81+7.5*((i-(1:cb))+0.474)-17.5*(1+((i-(1:cb))+0.474).^2).^0.5)));
        set(gca,'ylim',[-80 5])
        title('Spreading function for 10th band'); ylabel('dB'); xlabel('Bark')
    end

    % figure out number of fft frames
    frames = 0;
    idx = p.fft_size;
    while idx <= length(wav), 
        frames = frames + 1; 
        idx = idx + p.hopsize;
    end

    wav = wav * (10^(p.dB_max/20)); % rescale to dB max (default is 96dB = 2^16)

    dlinear = zeros(p.fft_size/2+1,frames); % data from fft (linear freq scale)
    sone  = zeros(cb,frames);              % data after bark scale

    idx = 1:p.fft_size;
    w = hann(p.fft_size);
    for i=1:frames, % FFT
        X = fft(wav(idx).*w,p.fft_size);
        dlinear(:,i) = abs(X(1:p.fft_size/2+1)/sum(w)*2).^2; % normalized powerspectrum

        idx = idx + p.hopsize;
    end

    if p.do_visu,
        figure; % with 6 rows depicting intermediate results of 5 processing steps

        hs(1) = subplot(6,1,1); % audio signal
        plot(wav,'k'); axis tight;
        set(gca,'xtick',[],'ytick',[]); ylabel('PCM')

        hs(2) = subplot(6,1,2); % spectrogram
        tmp.dlinear = dlinear; tmp.dlinear(tmp.dlinear<1)=1; % for dB 
        imagesc(10*log10(tmp.dlinear)); set(gca,'ydir','normal','xtick',[],'ytick',[]);
        ylabel('FFT')
    end

    dlinear = repmat(c.W_Adb',1,size(dlinear,2)).*dlinear; % outer ear

    if p.do_visu,
        hs(3) = subplot(6,1,3); % outer ear
        tmp.dlinear = dlinear; tmp.dlinear(tmp.dlinear<1)=1; % for dB
        imagesc(10*log10(tmp.dlinear)); set(gca,'ydir','normal','xtick',[],'ytick',[])
        ylabel('Outer Ear')
    end

    k = 1;
    for i=1:cb, % group into bark bands
        idx = find(c.fft_freq(k:end)<=c.bark_upper(i));
        idx = idx + k-1;
        sone(i,1:frames) = sum(dlinear(idx,1:frames),1);
        k = max(idx)+1;
    end

    if p.do_visu,
        hs(4) = subplot(6,1,4); % bark
        tmp.sone = sone; tmp.sone(tmp.sone<1) = 1; % for dB
        imagesc(10*log10(tmp.sone)); set(gca,'ydir','normal','xtick',[],'ytick',[])
        ylabel('Bark Scale')
    end

    if p.do_spread,
        sone = c.spread*sone; % spectral masking
    end

    sone(sone<1) = 1; sone = 10*log10(sone); % dB

    if p.do_visu,
        hs(5) = subplot(6,1,5); % masking
        imagesc(sone); set(gca,'ydir','normal','xtick',[],'ytick',[])
        ylabel('Masking')
    end

    % bladon and lindblom, 1981, JASA, modelling the judment of vowel quality differences
    if p.do_sone,
        idx = sone>=40;
        sone(idx) = 2.^((sone(idx)-40)/10);
        sone(~idx) = (sone(~idx)/40).^2.642;
    end

    if p.do_visu,
        hs(6) = subplot(6,1,6); % sone
        imagesc(sone); set(gca,'ydir','normal','xtick',[],'ytick',[])
        ylabel('Sone');
        for i=1:length(hs),
            pos = get(hs(i),'position');
            set(hs(i),'position',pos.*[1 1 1 1]+[-.05 -.08+0.02*(6-i) 0.05 0.03])
        end
    end

    if nargout >= 2, % Ntot requested
        Ntot = zeros(size(sone,2),1);
        not_idx = logical(ones(1,size(sone,1)));
        for i=1:size(sone,2),
            [maxi idx] = max(sone(:,i));
            not_idx(idx) = logical(0);
            Ntot(i) = maxi + 0.15*sum(sone(not_idx,i));
        end
    end

    if nargout == 0,
        sone = 'done'; % if only testing dont flood command window with numbers
    end

end