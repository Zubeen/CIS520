function [ gmm LL ] = gmm_em(X, K, varargin)
% Fits a mixture of K axis-aligned gaussians to data X.
%
% Usage:
%
%   [gmm L] = gmm_em(X, K, ...)
%
% gmm - structure containing model parameters. L is log likelihood.
%
% See M file for info on optional parameters.
%
% David Weiss, 2008-2009

defaults.maxiter = 200;
defaults.tol = 1e-5; % Convergence tolerance
defaults.burnin = 1; % # of mean fits before variance is fit
defaults.verbose = true; % whether to output

options = propval(varargin, defaults);

N = rows(X);
D = cols(X);

% Initialize means to K different example points
idx = randperm(N);
mu = X(idx(1:K), :);

% Initialize variance to be variances of random portions of the data
for k = 1:K
    idx = randperm(N);
    m = ceil(N / K);
    sigmasq(k,:) = var(X(idx(1:m),:));
end

if any(var(X) == 0)
    error(['Cannot run GMM: input has zero variance in at least one ' ...
           'dimension.']);
end

% Initialize equal prior probabilities
theta = log(ones(K,1) ./ K);

% Will hold logposterior probalitity that Z_n = k
gamma = zeros(N,K);
LL = nan;
for t = 1:options.maxiter


    try

        % E step: gamma_ik = posterior prob. of z_i = k
            
        % Compute unnormalized log posterior (i.e. likelihood)
        for k = 1:K  
            gamma(:,k) = normpdfln( X', mu(k,:)', [], diag(sigmasq(k,:)) ) ...
                + theta(k);
        end
        % Normalize in log domain:
        gamma = gamma - repmat(logsumexp(gamma,2), 1, K);
        
        % M step: compute means (mu) and variances (sigma)
        
        egammasum = exp(logsumexp(gamma)); % compute sum without underflow
        egamma = exp(gamma); % precompute exp(gamma)
        
        for k = 1:K
            egamma_k = repmat(egamma(:,k), 1, D);
            
            % Compute weighted mean 
            mu(k,:) = sum(X .* egamma_k ) ./ egammasum(k); 

            theta(k) = logsumexp(gamma(:,k)) - log(N);
            
            % Compute weighted variance only if the means have begun to settle
            if t > options.burnin
                sigmasq(k,:) = sum( [X - repmat(mu(k,:), N, 1)].^2 .* ...
                                    egamma_k ) ./ egammasum(k);
            end
        end

        % Bookkeeping step:
    
        % Compute log likelihood for convergence testing
        L(t) = 0;
        for k = 1:K
            % compute to marginalize out Z for the log likelihood
            cl(k,:) = normpdfln( X', mu(k,:)', [], diag(sigmasq(k,:)) ) +...
                      theta(k);
            
        end
        L(t) = sum(logsumexp(cl, 2));

        LL = L(t);
        
        if isnan(L(t))
            error('isnan', 'NaN likelihood detected.');
        end
        
        % Store this as the latest model
        gmm = bundle(gamma, theta, mu, sigmasq, options, L);        
    catch
        gmm.options.error = lasterror;
        warning(['Error ''%s'' occured -- did not converge. See ' ...
                 'options.error'], gmm.options.error.identifier);
        LL = nan;
        break;
    end
    
    % Show some progress if specified
    if options.verbose
        if t <= options.burnin
            fprintf('\tt = %d: L = %g (no variance update)\n', t, L(t));
        else
            fprintf('\tt = %d: L = %g\n', t, L(t));            
        end
    end
    
    % Check for convergence
    if t > options.burnin
        d = abs([L(t) - L(t-1)]) ./ abs(L(t-1));
        if d < options.tol
            LL = L(t);
            break;
        end
    end

end



