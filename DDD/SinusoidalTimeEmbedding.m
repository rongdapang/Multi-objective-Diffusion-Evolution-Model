function embedding = SinusoidalTimeEmbedding(t, embedding_dim)
% SinusoidalTimeEmbedding - Generate sinusoidal time embeddings
%
% This function generates sinusoidal time embeddings for diffusion models,
% similar to the positional encodings used in Transformers.
%
% Input:
%   t - Time values (normalized to [0, 1] or integers)
%   embedding_dim - Dimension of the embedding (should be even)
%
% Output:
%   embedding - Sinusoidal embeddings of size [length(t), embedding_dim]
%
% Reference:
%   Vaswani et al. "Attention Is All You Need" NeurIPS 2017

    if nargin < 2
        embedding_dim = 128;
    end
    
    % Ensure embedding_dim is even
    if mod(embedding_dim, 2) ~= 0
        embedding_dim = embedding_dim + 1;
    end
    
    n = length(t);
    embedding = zeros(n, embedding_dim);
    
    % Position encodings
    position = (1:embedding_dim/2)';
    div_term = exp(-log(10000) * (2 * (position - 1)) / embedding_dim);
    
    for i = 1:n
        % Sine for even indices
        embedding(i, 1:2:embedding_dim) = sin(t(i) * div_term);
        % Cosine for odd indices
        embedding(i, 2:2:embedding_dim) = cos(t(i) * div_term);
    end
end
