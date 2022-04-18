%% Mode mixing and Leakage index (ML-index)
% Quantifying mode mixing and leakage in multivariate empirical mode decomposition and application in motor imagery–based brain-computer interface system

%% high gamma dataset
raw_path = 'F:\Motor_Imagery_dataset\high-gamma-dataset_MEMD\raw\';
org_path = 'F:\Motor_Imagery_dataset\high-gamma-dataset_MEMD\Origin_MEMD\';
na_path = 'F:\Motor_Imagery_dataset\high-gamma-dataset_MEMD\NA_MEMD\';
nn_path = 'F:\Motor_Imagery_dataset\high-gamma-dataset_MEMD\NN_MEMD\';
nann_path = 'F:\Motor_Imagery_dataset\high-gamma-dataset_MEMD\NANN_MEMD\';

f1 = 8; f2 = 31; % mu = 8~12; beta = 18~25 or 8~15; 16~31
fr_b = [f1 f2];
fs = 160;

E_org = []; E_na = []; E_nn = []; E_nann = [];
%% Training data
X_tr = dir([raw_path '*train.mat']);
for i = 1:size(X_tr,1)
    disp(['training : ' num2str(i)])
    load([X_tr(i).folder '\' X_tr(i).name]);
    Org_tr = dir([org_path '\' X_tr(i).name(1:end-4) '*.mat']);
    Na_tr = dir([na_path '\'  X_tr(i).name(1:end-4) '*.mat']);
    Nn_tr = dir([nn_path '\'  X_tr(i).name(1:end-4) '*0406.mat']);
    Nann_tr = dir([nann_path '\'  X_tr(i).name(1:end-4) '*0406.mat']);
    for j = 1:4
        raw = d_raw(20*(j-1)+1:20*j,:,:,:,:);
        Org = load([Org_tr(j).folder '\' Org_tr(j).name]);
        Na = load([Na_tr(j).folder '\' Na_tr(j).name]);
        Nn = load([Nn_tr(j).folder '\' Nn_tr(j).name]);
        Nann = load([Nann_tr(j).folder '\' Nann_tr(j).name]);
        
        % Org ML-index
        org = Org.d_MEMD;
        na = Na.d_MEMD;
        nn = permute(Nn.imf, [1 2 4 3 5]);
        nann = permute(Nann.imf, [1 2 4 3 5]);
        
        for clas = 1:size(raw,5)
            for trial = 1:size(raw,1)
                E_org_ch = []; E_na_ch = []; E_nn_ch = []; E_nann_ch = [];
                for ch = 1:size(raw,2)
                    raw_sig = squeeze(raw(trial, ch, :, :, clas));
                    raw_sig = raw_sig-mean(raw_sig);
                    n = length(raw_sig); f = (0:n-1)*(fs/n);
                    ft_raw = fft(raw_sig);
                    raw_pwr = abs(ft_raw).^2/n;
                    [max_pwr, f_phi]= max(raw_pwr);
                    fr_b(3) = f(f_phi);
                    
                    diff_org1 = 0; C_org = [];  diff_na1 = 0; C_na = []; diff_nn1 = 0; C_nn = [];  diff_nann1 = 0; C_nann = [];
                    for f_i = 1:2
                        f0 = fr_b(f_i);
                        for imf = 1:size(org,3)
                            %% For original MEMD
                            org_sig = squeeze(org(trial, ch, imf, :, clas));
                            ft_org = fft(org_sig);
                            org_pwr = abs(ft_org).^2/n;
                            diff_org1 =diff_org1 + abs(raw_pwr(find(f==f0)) - org_pwr(find(f==f0)));
                            C_org{f_i, imf} = org_pwr(find(f==f0));
                            C_org{3, imf} = org_pwr(find(f==f(f_phi)));
                            %% for NA-MEMD
                            na_sig = squeeze(na(trial, ch, imf, :, clas));
                            ft_na = fft(na_sig);
                            na_pwr = abs(ft_na).^2/n;
                            diff_na1 =diff_na1 + abs(raw_pwr(find(f==f0)) - na_pwr(find(f==f0)));
                            C_na{f_i, imf} = na_pwr(find(f==f0));
                            C_na{3, imf} = na_pwr(find(f==f(f_phi)));
                            %% for NN-MEMD
                            nn_sig = squeeze(nn(trial, ch, imf, :, clas));
                            ft_nn = fft(nn_sig);
                            nn_pwr = abs(ft_nn).^2/n;
                            diff_nn1 =diff_nn1 + abs(raw_pwr(find(f==f0)) - nn_pwr(find(f==f0)));
                            C_nn{f_i, imf} = nn_pwr(find(f==f0));
                            C_nn{3, imf} = nn_pwr(find(f==f(f_phi)));
                            %% for NANN-MEMD
                            nann_sig = squeeze(nann(trial, ch, imf, :, clas));
                            ft_nann = fft(nann_sig);
                            nann_pwr = abs(ft_nann).^2/n;
                            diff_nann1 =diff_nann1 + abs(raw_pwr(find(f==f0)) - nann_pwr(find(f==f0)));
                            C_nann{f_i, imf} = nann_pwr(find(f==f0));
                            C_nann{3, imf} = nann_pwr(find(f==f(f_phi)));
                        end
                    end
                    diff_org2 = 0; diff_na2 = 0; diff_nn2 = 0; diff_nann2 = 0;
                    for f_i = 1:3
                        diff_org2 = diff_org2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_org(f_i,:))]);
                        diff_na2 = diff_na2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_na(f_i,:))]);
                        diff_nn2 = diff_nn2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_nn(f_i,:))]);
                        diff_nann2 = diff_nann2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_nann(f_i,:))]);
                    end
                    E_org_ch = [E_org_ch; (diff_org1 + diff_org2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                    E_na_ch = [E_na_ch; (diff_na1 + diff_na2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                    E_nn_ch = [E_nn_ch; (diff_nn1 + diff_nn2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                    E_nann_ch = [E_nann_ch; (diff_nann1 + diff_nann2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                end
                E_org = [ E_org ; mean(E_org_ch)];
                E_na = [ E_na ; mean(E_na_ch)];
                E_nn = [ E_nn ; mean(E_nn_ch)];
                E_nann = [ E_nann ; mean(E_nann_ch)];
            end
        end
    end
end
X_ts = dir([raw_path '*test.mat']);
for i = 1:size(X_ts,1)
    disp(['test : ' num2str(i)])
    load([X_ts(i).folder '\' X_ts(i).name]);
    Org_ts = dir([org_path '\' X_ts(i).name(1:end-4) '*.mat']);
    Na_ts = dir([na_path '\'  X_ts(i).name(1:end-4) '*.mat']);
    Nn_ts = dir([nn_path '\'  X_ts(i).name(1:end-4) '*0406.mat']);
    Nann_ts = dir([nann_path '\'  X_ts(i).name(1:end-4) '*0406.mat']);
    for j = 1:2
        raw = d_raw(20*(j-1)+1:20*j,:,:,:,:);
        Org = load([Org_ts(j).folder '\' Org_ts(j).name]);
        Na = load([Na_ts(j).folder '\' Na_ts(j).name]);
        Nn = load([Nn_ts(j).folder '\' Nn_ts(j).name]);
        Nann = load([Nann_ts(j).folder '\' Nann_ts(j).name]);
        
        % Org ML-index
        org = Org.d_MEMD;
        na = Na.d_MEMD;
        nn = permute(Nn.imf, [1 2 4 3 5]);
        nann = permute(Nann.imf, [1 2 4 3 5]);
        
        for clas = 1:size(raw,5)
            for trial = 1:size(raw,1)
                E_org_ch = []; E_na_ch = []; E_nn_ch = []; E_nann_ch = [];
                for ch = 1:size(raw,2)
                    raw_sig = squeeze(raw(trial, ch, :, :, clas));
                    raw_sig = raw_sig-mean(raw_sig);
                    n = length(raw_sig); f = (0:n-1)*(fs/n);
                    ft_raw = fft(raw_sig);
                    raw_pwr = abs(ft_raw).^2/n;
                    [max_pwr, f_phi]= max(raw_pwr);
                    fr_b(3) = f(f_phi);
                    
                    diff_org1 = 0; C_org = [];  diff_na1 = 0; C_na = []; diff_nn1 = 0; C_nn = [];  diff_nann1 = 0; C_nann = [];
                    for f_i = 1:2
                        f0 = fr_b(f_i);
                        for imf = 1:size(org,3)
                            %% For original MEMD
                            org_sig = squeeze(org(trial, ch, imf, :, clas));
                            ft_org = fft(org_sig);
                            org_pwr = abs(ft_org).^2/n;
                            diff_org1 =diff_org1 + abs(raw_pwr(find(f==f0)) - org_pwr(find(f==f0)));
                            C_org{f_i, imf} = org_pwr(find(f==f0));
                            C_org{3, imf} = org_pwr(find(f==f(f_phi)));
                            %% for NA-MEMD
                            na_sig = squeeze(na(trial, ch, imf, :, clas));
                            ft_na = fft(na_sig);
                            na_pwr = abs(ft_na).^2/n;
                            diff_na1 =diff_na1 + abs(raw_pwr(find(f==f0)) - na_pwr(find(f==f0)));
                            C_na{f_i, imf} = na_pwr(find(f==f0));
                            C_na{3, imf} = na_pwr(find(f==f(f_phi)));
                            %% for NN-MEMD
                            nn_sig = squeeze(nn(trial, ch, imf, :, clas));
                            ft_nn = fft(nn_sig);
                            nn_pwr = abs(ft_nn).^2/n;
                            diff_nn1 =diff_nn1 + abs(raw_pwr(find(f==f0)) - nn_pwr(find(f==f0)));
                            C_nn{f_i, imf} = nn_pwr(find(f==f0));
                            C_nn{3, imf} = nn_pwr(find(f==f(f_phi)));
                            %% for NANN-MEMD
                            nann_sig = squeeze(nann(trial, ch, imf, :, clas));
                            ft_nann = fft(nann_sig);
                            nann_pwr = abs(ft_nann).^2/n;
                            diff_nann1 =diff_nann1 + abs(raw_pwr(find(f==f0)) - nann_pwr(find(f==f0)));
                            C_nann{f_i, imf} = nann_pwr(find(f==f0));
                            C_nann{3, imf} = nann_pwr(find(f==f(f_phi)));
                        end
                    end
                    diff_org2 = 0; diff_na2 = 0; diff_nn2 = 0; diff_nann2 = 0;
                    for f_i = 1:3
                        diff_org2 = diff_org2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_org(f_i,:))]);
                        diff_na2 = diff_na2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_na(f_i,:))]);
                        diff_nn2 = diff_nn2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_nn(f_i,:))]);
                        diff_nann2 = diff_nann2 + raw_pwr(find(f==fr_b(f_i))) - max([cell2mat(C_nann(f_i,:))]);
                    end
                    E_org_ch = [E_org_ch; (diff_org1 + diff_org2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                    E_na_ch = [E_na_ch; (diff_na1 + diff_na2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                    E_nn_ch = [E_nn_ch; (diff_nn1 + diff_nn2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                    E_nann_ch = [E_nann_ch; (diff_nann1 + diff_nann2)/mean(raw_pwr(find(f==fr_b(1)):find(f==fr_b(2))))];
                end
                E_org = [ E_org ; mean(E_org_ch)];
                E_na = [ E_na ; mean(E_na_ch)];
                E_nn = [ E_nn ; mean(E_nn_ch)];
                E_nann = [ E_nann ; mean(E_nann_ch)];
            end
        end
    end
end
[mean([E_org E_na E_nn E_nann]); std([E_org E_na E_nn E_nann])]
boxplot([E_org E_na E_nn E_nann])
